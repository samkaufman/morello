use morello::codegen::{CodeGen, CpuCodeGenThreadStyle};
use morello::common::{DimSize, Dtype};
use morello::db::RocksDatabase;
use morello::layout::{col_major, row_major, Layout, PhysDim};
use morello::lspec;
use morello::pprint::{pprint, ImplPrintStyle};
use morello::scheduling_sugar::{SchedulingSugar, Subschedule};
use morello::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
use morello::target::{
    CpuKernel,
    CpuMemoryLevel::{self, GL},
    Target, X86Target,
};
use morello::tensorspec::TensorSpecAux;
use morello::utils::ToWriteFmt;

use clap::Parser;
use nonzero::nonzero as nz;
use smallvec::smallvec;
use std::io;
use std::path;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, short)]
    db: Option<path::PathBuf>,
}

fn main() {
    let args = Args::parse();
    let db = RocksDatabase::try_new(args.db.as_deref(), true, 1).unwrap();

    const M: DimSize = nz!(1u32);
    const K: DimSize = nz!(2048u32);
    const N: DimSize = nz!(16384u32);

    let spec = Spec::<X86Target>(
        lspec!(Matmul(
            [M, K, N],
            (bf16, GL, row_major(2)),
            (bf16, GL, col_major(2)),
            (f32, GL, row_major(2))
        )),
        X86Target::max_mem(),
    );
    println!("Logical Spec: {}", spec.0);

    // Manually schedule the matrix multiplication.
    let interleaved = Layout::new(smallvec![
        (0, PhysDim::Dynamic),
        (1, PhysDim::Dynamic),
        (1, PhysDim::Interleaved(nz!(16u32)))
    ]);

    let implementation = spec
        .cast(
            0,
            Dtype::Float32,
            CpuMemoryLevel::L1,
            interleaved.clone(),
            None,
        )
        .subschedule(&[0], &|z| {
            z.tile_out(&[1, 16], false)
                .move_param(0, CpuMemoryLevel::L1, row_major(2), None)
                .move_param(0, CpuMemoryLevel::VRF, row_major(2), Some(nz!(16u32)))
                .subschedule(&[0], &|z| z.place(CpuKernel::VectorAssign))
                .subschedule(&[1], &|z| {
                    z.move_param(1, CpuMemoryLevel::VRF, interleaved.clone(), Some(nz!(8u32)))
                        .subschedule(&[0], &|z| z.place(CpuKernel::VectorInterleaveBf16F32))
                        .subschedule(&[1], &|z| {
                            z.tile_out(&[1, 8], false).place(CpuKernel::VectorAssign)
                        })
                })
        })
        .subschedule(&[1], &|body| {
            body.tile_out(&[1, 128], true)
                .tile_out(&[1, 1], false)
                .move_param(2, CpuMemoryLevel::L1, row_major(2), None)
                .move_param(2, CpuMemoryLevel::RF, row_major(2), None)
                .subschedule(&[0], &|z| z.to_accum())
                .subschedule(&[0, 0], &|z| z.place(CpuKernel::MemsetZero))
                .subschedule(&[0, 1], &|body| {
                    body.move_param(1, CpuMemoryLevel::L1, col_major(2), None)
                        .place(CpuKernel::DotProductLoopF32InterleavedBf16F32)
                })
                .subschedule(&[1], &|body| {
                    body.move_param(0, CpuMemoryLevel::L1, col_major(2), None)
                        .subschedule(&[0], &|z| z.place(CpuKernel::ValueAssign))
                        .subschedule(&[1], &|z| z.place(CpuKernel::ValueAssign))
                })
        });

    // Drop the DB to flush early.
    drop(db);

    println!("\nImpl resulting from manual scheduling:");
    pprint(&implementation, ImplPrintStyle::Full);

    println!("\nThe above Impl lowered to C:");
    implementation
        .emit_ext(
            false,
            None,
            CpuCodeGenThreadStyle::Highway,
            &mut ToWriteFmt(io::stdout()),
        )
        .unwrap();

    // If the verification flag is set, let's additionally double-check that the lowered
    // code builds and produces the correct results.
    #[cfg(feature = "verification")]
    {
        let artifact = implementation.build(false).unwrap();
        if !artifact.check_correctness(&spec) {
            panic!("Generated code returned incorrect output");
        }
    }

    // And benchmark!
    let result = implementation.bench(3, None).unwrap();
    let inner_loop_runtime = result.best_inner_loop_runtime();
    let kernel_runtime = inner_loop_runtime / result.inner_loop_iterations;
    println!("\nkernel runtime: {:.8}s", kernel_runtime.as_secs_f32());
    println!("loop runtime: {}ns", inner_loop_runtime.as_nanos());
}
