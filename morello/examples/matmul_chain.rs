use morello::codegen::CodeGen;
use morello::common::Dtype;
use morello::cost::Cost;
use morello::db::FilesDatabase;
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
use std::path::Path;

use nonzero::nonzero as nz;

fn main() {
    let basics0 = PrimitiveBasics {
        typ: PrimitiveSpecType::Matmul { accum: false },
        spec_shape: shape![1024, 1024, 1024],
        dtypes: vec![Dtype::Float32, Dtype::Float32, Dtype::Float32],
    };
    let basics1 = PrimitiveBasics {
        typ: PrimitiveSpecType::Matmul { accum: false },
        spec_shape: shape![1024, 1024, 1024],
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

    let db = FilesDatabase::new(
        Some(Path::new("./matmul_chain.db")),
        true,
        1,
        2usize.pow(22),
        2,
        None,
    );
    let imp = spec
        .tile_out(&[128, 128])
        .bufferize(0, GL, row_major(2), None)
        .subschedule(&[0], |s| schedule_matmul(&db, s))
        .subschedule(&[1], |s| schedule_matmul(&db, s));
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
    const ITERS: u32 = 5;
    let result = imp
        .bench(ITERS, None)
        .unwrap_or_else(|e| panic!("Failed to benchmark: {}", e));
    let kernel_runtime =
        (result.best_inner_loop_runtime() / result.inner_loop_iterations).as_secs_f64();
    let throughput =
        result.inner_loop_iterations as f64 / result.best_inner_loop_runtime().as_secs_f64();
    println!("\n// cost: {}", Cost::from_impl(&imp).main);
    println!("// kernel runtime: {kernel_runtime:.4}s ({throughput:.2}/sec)",);
}

fn schedule_matmul(db: &FilesDatabase, spec: &Spec<X86Target>) -> ImplNode<X86Target> {
    let mat1_pack_size = nz!(16u32);
    let layout_b = Layout::new(vec![
        (1, PhysDim::Dynamic),
        (0, PhysDim::Dynamic),
        (1, PhysDim::Packed(mat1_pack_size)),
    ]);
    spec.to_accum()
        .subschedule(&[0], |zero_spec| {
            zero_spec.tile_out(&[1, 4]).synthesize(db, None)
        })
        .split(128)
        .move_param(1, GL, layout_b.clone(), None)
        .subschedule(&[1, 0], |pack_b| {
            // TODO: This stinks. Use vectors at least.
            pack_b
                .tile_out(&[1, 1])
                .move_param(0, L1, row_major(2), None)
                .move_param(1, L1, row_major(2), None)
                .move_param(0, RF, row_major(2), None)
                .subschedule(&[0], |m0| m0.place(CpuKernel::ValueAssign))
                .subschedule(&[1], |m0| m0.place(CpuKernel::ValueAssign))
        })
        .tile_out(&[4, 16])
        .move_param(0, L1, row_major(2), None)
        .move_param(1, L1, layout_b.clone(), None)
        .move_param(2, L1, row_major(2), None)
        .move_param(2, VRF, row_major(2), Some(nz!(8u32)))
        .subschedule(&[1, 1, 0], |m| {
            m.tile_out(&[1, 8]).place(CpuKernel::VectorAssign)
        })
        .subschedule(&[1, 1, 2], |m| {
            m.tile_out(&[1, 8]).place(CpuKernel::VectorAssign)
        })
        .split(1)
        .tile_out(&[1, 16])
        .move_param(1, VRF, layout_b.clone(), Some(nz!(8u32)))
        .subschedule(&[1, 1, 1, 0], |m| m.place(CpuKernel::VectorAssign))
        .subschedule(&[1, 1, 1, 1], |m| m.place(CpuKernel::BroadcastVecMultAdd))
}
