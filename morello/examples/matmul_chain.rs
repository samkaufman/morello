use morello::codegen::CodeGen;
use morello::common::Dtype;
use morello::cost::Cost;
use morello::imp::ImplNode;
use morello::layout;
use morello::layout::{row_major, Layout};
use morello::pprint::ImplPrintStyle;
use morello::scheduling_sugar::{SchedulingSugar, Subschedule};
use morello::shape;
use morello::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
use morello::target::CpuKernel;
use morello::target::{
    Avx2Target,
    CpuMemoryLevel::{GL, L1, RF, VRF},
    Target,
};
use morello::tensorspec::TensorSpecAux;
use morello::utils::ToWriteFmt;

use std::io;

const BATCH: u32 = 1;

fn main() {
    let basics0 = PrimitiveBasics {
        typ: PrimitiveSpecType::Matmul { accum: false },
        spec_shape: shape![BATCH, 2048, 2048, 2048],
        dtypes: vec![Dtype::Float32, Dtype::Float32, Dtype::Float32],
    };
    let basics1 = PrimitiveBasics {
        typ: PrimitiveSpecType::Softmax { scan_dim: 2 },
        spec_shape: shape![BATCH, 2048, 2048],
        dtypes: vec![Dtype::Float32, Dtype::Float32],
    };
    let basics2 = PrimitiveBasics {
        typ: PrimitiveSpecType::Matmul { accum: false },
        spec_shape: shape![BATCH, 2048, 2048, 2048],
        dtypes: vec![Dtype::Float32, Dtype::Float32, Dtype::Float32],
    };
    let tensors_shape = shape![BATCH, 2048, 2048];
    let aux = TensorSpecAux {
        level: GL,
        layout: row_major(&tensors_shape),
        vector_size: None,
    };

    let mut spec = Spec::<Avx2Target>(
        LogicalSpec::Compose {
            components: vec![basics2, basics1, basics0],
            operand_auxes: vec![aux.clone(), aux.clone(), aux.clone(), aux],
            serial_only: true,
        },
        Avx2Target::max_mem(),
    );
    spec.canonicalize().unwrap();

    let mut imp = spec.bufferize(0, GL, row_major, None);
    if BATCH > 1 {
        imp = imp.subschedule(&[0], |tail_compose| tail_compose.tile_out(&[1, 2048, 2048]));
    }
    imp = imp
        .subschedule(&[0], |tail_compose| {
            tail_compose
                .bufferize(0, GL, row_major, None)
                .subschedule(&[0], |initial_matmul| {
                    initial_matmul
                        .to_accum()
                        .subschedule(&[0], schedule_zero)
                        .subschedule(&[1], schedule_matmulaccum)
                })
                .subschedule(&[1], schedule_softmax)
        })
        .subschedule(&[1], |head_matmul| {
            head_matmul
                .to_accum()
                .subschedule(&[0], schedule_zero)
                .subschedule(&[1], schedule_matmulaccum)
        });
    imp.emit(
        false,
        Some(ImplPrintStyle::Compact),
        &mut ToWriteFmt(io::stdout()),
    )
    .unwrap_or_else(|e| panic!("Failed to generate code: {e}"));

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
                panic!("Failed to build generated code: {e}");
            }
        }
    }

    // Benchmark.
    const ITERS: u32 = 10;
    let result = imp
        .bench(ITERS, None)
        .unwrap_or_else(|e| panic!("Failed to benchmark: {e}"));
    let kernel_runtime =
        (result.best_inner_loop_runtime() / result.inner_loop_iterations).as_secs_f64();
    let throughput =
        result.inner_loop_iterations as f64 / result.best_inner_loop_runtime().as_secs_f64();
    println!("\n// cost: {}", Cost::from_impl(&imp).main);
    println!("// kernel runtime: {kernel_runtime:.4}s ({throughput:.2}/sec)");
}

fn schedule_matmulaccum(spec: &Spec<Avx2Target>) -> ImplNode<Avx2Target> {
    spec.split(128)
        .move_relayout(1, GL, layout_b(), None)
        .tile_out(&[1, 128, 1024])
        .tile_out(&[1, 4, 16])
        .move_param(0, L1)
        .move_param(1, L1)
        .move_param(2, L1)
        .move_vrf(2, VRF, 8)
        .split(1)
        .tile_out(&[1, 1, 16])
        .move_vrf(1, VRF, 8)
        .select(CpuKernel::BroadcastVecMultAdd)
        .subschedule(&[0], |pack_b| {
            // TODO: This stinks. Use vectors at least.
            pack_b
                .tile_out(&[1, 1, 1])
                .move_param(0, L1)
                .move_param(1, L1)
                .move_param(0, RF)
                .subschedule(&[0], |m0| m0.select(CpuKernel::Assign))
                .subschedule(&[1], |m0| m0.select(CpuKernel::Assign))
        })
        .subschedule(&[1, 0], |m| {
            m.tile_out(&[1, 1, 8]).select(CpuKernel::Assign)
        })
        .subschedule(&[1, 1, 0], |m| {
            m.tile_out(&[1, 1, 8]).select(CpuKernel::Assign)
        })
        .subschedule(&[1, 2], |m| {
            m.tile_out(&[1, 1, 8]).select(CpuKernel::Assign)
        })
}

fn schedule_softmax(spec: &Spec<Avx2Target>) -> ImplNode<Avx2Target> {
    use morello::db::FilesDatabase;
    use morello::target::CpuMemoryLevel::{GL, L1, RF, VRF};

    let db = FilesDatabase::new::<Avx2Target>(None, true, 1, 10_000, 1);

    spec.tile_out(&[1, 1, 2048])
        .to_softmax_parts(GL, row_major, None, GL, row_major, None)
        .subschedule(&[0], |subspec| {
            subspec.to_max_and_unscaled(GL, row_major, None)
        })
        .subschedule(&[0, 0], |subspec| {
            subspec.to_accum().split(1).synthesize(&db)
        })
        .subschedule(&[0, 0, 0], |s| s.synthesize(&db))
        .subschedule(&[0, 1], |subspec| subspec.to_accum())
        .subschedule(&[0, 1, 0], |subspec| subspec.synthesize(&db))
        // SoftmaxDenominatorAndUnscaledFromMaxAccum
        .subschedule(&[0, 1, 1], |subspec| {
            subspec
                .tile_out(&[1, 1, 16])
                .move_param(0, L1)
                .move_param(1, L1)
                .move_param(2, L1)
                .move_param(3, L1)
                .move_param(1, RF)
                .move_param(2, RF)
                .subschedule(&[1, 1], |s| {
                    s.move_relayout(0, VRF, row_major, Some(8))
                        .move_relayout(3, VRF, row_major, Some(8))
                        .subschedule(&[0], |m| m.synthesize(&db))
                        .subschedule(&[1, 1], |m| m.synthesize(&db))
                        .subschedule(&[1, 0], |m| {
                            m.select(CpuKernel::VectorSoftmaxDenominatorAndUnscaledF32)
                        })
                })
                .subschedule(&[0], |m| m.synthesize(&db))
                .subschedule(&[1, 0], |m| m.synthesize(&db))
                .subschedule(&[1, 2], |m| m.synthesize(&db))
        })
        // DivideVec
        .subschedule(&[1], |subspec| {
            subspec
                .tile_out(&[1, 1, 8])
                .broadcast_first(VRF, row_major, Some(8))
                .subschedule(&[0], |broadcast| {
                    broadcast
                        .move_relayout(0, L1, row_major, None)
                        .move_relayout(0, RF, row_major, None)
                        .synthesize(&db)
                        .subschedule(&[0], |s| s.synthesize(&db))
                })
                .subschedule(&[1], |d| d.synthesize(&db))
        })
}

fn schedule_zero(spec: &Spec<Avx2Target>) -> ImplNode<Avx2Target> {
    spec.tile_out(&[1, 32, 1])
        .move_param(0, L1)
        .tile_out(&[1, 16, 1])
        .move_param(0, RF)
        .subschedule(&[0], |s| {
            s.tile_out(&[1, 1, 1]).select(CpuKernel::MemsetZero)
        })
        .subschedule(&[1], |s| s.tile_out(&[1, 1, 1]).select(CpuKernel::Assign))
}

fn layout_b() -> Layout {
    layout![0, 2, 1, 2 p(16)]
}
