use morello::codegen::CodeGen;
use morello::common::Dtype;
use morello::cost::Cost;
use morello::imp::ImplNode;
use morello::layout::{row_major, Layout, PhysDim};
use morello::pprint::ImplPrintStyle;
use morello::scheduling_sugar::{SchedulingSugar, Subschedule};
use morello::shape;
use morello::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
use morello::target::CpuKernel;
use morello::target::{
    CpuMemoryLevel::{GL, L1, RF, VRF},
    Target, X86Target,
};
use morello::tensorspec::TensorSpecAux;
use morello::utils::ToWriteFmt;

use std::io;

use nonzero::nonzero as nz;

fn main() {
    let basics0 = PrimitiveBasics {
        typ: PrimitiveSpecType::Matmul { accum: false },
        spec_shape: shape![32, 2048, 2048, 2048],
        dtypes: vec![Dtype::Float32, Dtype::Float32, Dtype::Float32],
    };
    let basics1 = PrimitiveBasics {
        typ: PrimitiveSpecType::Softmax { scan_dim: 0 },
        spec_shape: shape![32, 2048, 2048],
        dtypes: vec![Dtype::Float32, Dtype::Float32],
    };
    let basics2 = PrimitiveBasics {
        typ: PrimitiveSpecType::OnePrefix,
        spec_shape: shape![2048, 2048],
        dtypes: vec![Dtype::Float32, Dtype::Float32],
    };
    let basics3 = PrimitiveBasics {
        typ: PrimitiveSpecType::Matmul { accum: false },
        spec_shape: shape![1, 2048, 2048, 2048],
        dtypes: vec![Dtype::Float32, Dtype::Float32, Dtype::Float32],
    };
    let aux = TensorSpecAux {
        contig: row_major(3).contiguous_full(),
        aligned: true,
        level: GL,
        layout: row_major(3),
        vector_size: None,
    };

    let mut spec = Spec::<X86Target>(
        LogicalSpec::Compose {
            components: vec![basics3, basics2, basics1, basics0],
            operand_auxes: vec![aux.clone(), aux.clone(), aux.clone(), aux],
            serial_only: true,
        },
        X86Target::max_mem(),
    );
    spec.canonicalize().unwrap();

    let imp = spec
        .to_accum()
        .split(1024)
        .bufferize(0, GL, row_major, None)
        .subschedule(&[0], schedule_zero)
        .subschedule(&[1, 0], |s| {
            s.to_accum()
                .subschedule(&[0], schedule_zero)
                .subschedule(&[1], schedule_matmulaccum)
        })
        .subschedule(&[1, 1], schedule_matmulaccum);
    imp.emit(
        false,
        Some(ImplPrintStyle::Compact),
        &mut ToWriteFmt(io::stdout()),
    )
    .unwrap_or_else(|e| panic!("Failed to generate code: {}", e));

    // If the verification flag is set, let's additionally double-check that the lowered
    // code builds and produces the correct results.
    #[cfg(feature = "verification")]
    {
        match imp.build(false) {
            Ok(artifact) => {
                if !artifact.check_correctness(&spec) {
                    panic!("Generated code returned incorrect output");
                }
            }
            Err(e) => {
                panic!("Failed to build generated code: {}", e);
            }
        }
    }

    // Benchmark.
    const ITERS: u32 = 10;
    let result = imp
        .bench(ITERS, None)
        .unwrap_or_else(|e| panic!("Failed to benchmark: {}", e));
    let kernel_runtime =
        (result.best_inner_loop_runtime() / result.inner_loop_iterations).as_secs_f64();
    let throughput =
        result.inner_loop_iterations as f64 / result.best_inner_loop_runtime().as_secs_f64();
    println!("\n// cost: {}", Cost::from_impl(&imp).main);
    println!("// kernel runtime: {kernel_runtime:.4}s ({throughput:.2}/sec)");
}

fn schedule_matmulaccum(spec: &Spec<X86Target>) -> ImplNode<X86Target> {
    spec.split(128)
        .move_param(1, GL, layout_b(), None)
        .subschedule(&[0], |pack_b| {
            // TODO: This stinks. Use vectors at least.
            pack_b
                .tile_out(&[1, 1, 1])
                .move_param(0, L1, row_major, None)
                .move_param(1, L1, row_major, None)
                .move_param(0, RF, row_major, None)
                .subschedule(&[0], |m0| m0.select(CpuKernel::ValueAssign))
                .subschedule(&[1], |m0| m0.select(CpuKernel::ValueAssign))
        })
        .tile_out(&[1, 128, 1024])
        .tile_out(&[1, 4, 16])
        .move_param(0, L1, row_major, None)
        .move_param(1, L1, layout_b(), None)
        .move_param(2, L1, row_major, None)
        .move_param(2, VRF, row_major, Some(nz!(8u32)))
        .subschedule(&[1, 0], |m| {
            m.tile_out(&[1, 1, 8]).select(CpuKernel::VectorAssign)
        })
        .subschedule(&[1, 2], |m| {
            m.tile_out(&[1, 1, 8]).select(CpuKernel::VectorAssign)
        })
        .split(1)
        .tile_out(&[1, 1, 16])
        .move_param(1, VRF, layout_b(), Some(nz!(8u32)))
        .subschedule(&[1, 1, 0], |m| m.select(CpuKernel::VectorAssign))
        .subschedule(&[1, 1, 1], |m| m.select(CpuKernel::BroadcastVecMultAdd))
}

fn schedule_softmax(spec: &Spec<X86Target>) -> ImplNode<X86Target> {
    use morello::common::{DimSize, Dtype};
    use morello::db::FilesDatabase;
    use morello::target::CpuMemoryLevel::{GL, L1, RF, VRF};

    const RANK: u8 = 2;
    const SIZE: DimSize = nz!(128u32);
    let layouts = [row_major(RANK), row_major(RANK)];

    let db = FilesDatabase::new(None, true, 1, 10_000, 1, None);

    spec.tile_out(&[1, SIZE.get()])
        .to_softmax_parts(GL, row_major(RANK), None, GL, row_major(RANK), None)
        .subschedule(&[0], |subspec| {
            subspec.to_max_and_unscaled(GL, row_major(RANK), None)
        })
        .subschedule(&[0, 0], |subspec| {
            subspec.to_accum().split(1).synthesize(&db, None)
        })
        .subschedule(&[0, 0, 0], |s| s.synthesize(&db, None))
        .subschedule(&[0, 1], |subspec| {
            subspec
                .to_accum()
                .subschedule(&[0], |s| s.synthesize(&db, None))
                .subschedule(&[1], |s| {
                    s.move_param(0, L1, row_major(2), None)
                        .move_param(1, L1, row_major(2), None)
                        .move_param(2, L1, row_major(2), None)
                        .move_param(3, L1, row_major(2), None)
                        .move_param(0, VRF, row_major(2), Some(nz!(8u32)))
                        .subschedule(&[0], |m| m.tile_out(&[1, 8]).synthesize(&db, None))
                        .move_param(1, RF, row_major(2), None)
                        .subschedule(&[1, 0], |m| m.synthesize(&db, None))
                        .move_param(2, RF, row_major(2), None)
                        .subschedule(&[1, 1, 0], |m| m.synthesize(&db, None))
                        .subschedule(&[1, 1, 2], |m| m.synthesize(&db, None))
                        .move_param(3, VRF, row_major(2), Some(nz!(8u32)))
                        .subschedule(&[1, 1, 1, 0], |m| m.synthesize(&db, None))
                        .subschedule(&[1, 1, 1, 1], |m| m.synthesize(&db, None))
                })
        })
        .subschedule(&[1], |subspec| {
            subspec
                .tile_out(&[1, 4])
                .broadcast_first(VRF, row_major(RANK), Some(nz!(4u32)))
                .subschedule(&[0], |broadcast| {
                    broadcast
                        .move_param(0, L1, row_major(2), None)
                        .move_param(0, RF, row_major(2), None)
                        .synthesize(&db, None)
                        .subschedule(&[0], |s| s.synthesize(&db, None))
                })
                .subschedule(&[1], |d| d.synthesize(&db, None))
        })
}

fn schedule_zero(spec: &Spec<X86Target>) -> ImplNode<X86Target> {
    spec.tile_out(&[1, 32, 1])
        .move_param(0, L1, row_major, None)
        .tile_out(&[1, 16, 1])
        .move_param(0, RF, row_major, None)
        .subschedule(&[0], |s| {
            s.tile_out(&[1, 1, 1]).select(CpuKernel::MemsetZero)
        })
        .subschedule(&[1], |s| {
            s.tile_out(&[1, 1, 1]).select(CpuKernel::ValueAssign)
        })
}

fn layout_b() -> Layout {
    let mat1_pack_size = nz!(16u32);
    Layout::new(vec![
        (0, PhysDim::Dynamic),
        (2, PhysDim::Dynamic),
        (1, PhysDim::Dynamic),
        (2, PhysDim::Packed(mat1_pack_size)),
    ])
}
