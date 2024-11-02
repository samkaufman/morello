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
        spec_shape: shape![2048, 2048, 2048],
        dtypes: vec![Dtype::Float32, Dtype::Float32, Dtype::Float32],
    };
    let basics1 = PrimitiveBasics {
        typ: PrimitiveSpecType::Matmul { accum: false },
        spec_shape: shape![2048, 2048, 2048],
        dtypes: vec![Dtype::Float32, Dtype::Float32, Dtype::Float32],
    };
    let aux = TensorSpecAux {
        contig: row_major(2).contiguous_full(),
        aligned: true,
        level: GL,
        layout: row_major(2),
        vector_size: None,
    };

    let mut spec = Spec::<X86Target>(
        LogicalSpec::Compose {
            components: vec![basics1, basics0],
            operand_auxes: vec![aux.clone(), aux.clone(), aux.clone(), aux],
            serial_only: true,
        },
        X86Target::max_mem(),
    );
    spec.canonicalize().unwrap();

    let imp = spec
        .to_accum()
        .split(1024)
        .bufferize(0, GL, row_major(2), None)
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
                .tile_out(&[1, 1])
                .move_param(0, L1, row_major(2), None)
                .move_param(1, L1, row_major(2), None)
                .move_param(0, RF, row_major(2), None)
                .subschedule(&[0], |m0| m0.place(CpuKernel::ValueAssign))
                .subschedule(&[1], |m0| m0.place(CpuKernel::ValueAssign))
        })
        .tile_out(&[128, 1024])
        .tile_out(&[4, 16])
        .move_param(0, L1, row_major(2), None)
        .move_param(1, L1, layout_b(), None)
        .move_param(2, L1, row_major(2), None)
        .move_param(2, VRF, row_major(2), Some(nz!(8u32)))
        .subschedule(&[1, 0], |m| {
            m.tile_out(&[1, 8]).place(CpuKernel::VectorAssign)
        })
        .subschedule(&[1, 2], |m| {
            m.tile_out(&[1, 8]).place(CpuKernel::VectorAssign)
        })
        .split(1)
        .tile_out(&[1, 16])
        .move_param(1, VRF, layout_b(), Some(nz!(8u32)))
        .subschedule(&[1, 1, 0], |m| m.place(CpuKernel::VectorAssign))
        .subschedule(&[1, 1, 1], |m| m.place(CpuKernel::BroadcastVecMultAdd))
}

fn schedule_zero(spec: &Spec<X86Target>) -> ImplNode<X86Target> {
    spec.tile_out(&[32, 1])
        .move_param(0, L1, row_major(2), None)
        .tile_out(&[16, 1])
        .move_param(0, RF, row_major(2), None)
        .subschedule(&[0], |s| s.tile_out(&[1, 1]).place(CpuKernel::MemsetZero))
        .subschedule(&[1], |s| s.tile_out(&[1, 1]).place(CpuKernel::ValueAssign))
}

fn layout_b() -> Layout {
    let mat1_pack_size = nz!(16u32);
    Layout::new(vec![
        (1, PhysDim::Dynamic),
        (0, PhysDim::Dynamic),
        (1, PhysDim::Packed(mat1_pack_size)),
    ])
}
