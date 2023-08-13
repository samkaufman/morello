use iai_callgrind::{black_box, main};
use smallvec::{smallvec, SmallVec};

use morello::common::Dtype;
use morello::layout::row_major;
use morello::spec::{LogicalSpec, PrimitiveAux, PrimitiveBasics, PrimitiveSpecType};
use morello::target::{CpuMemoryLevel, Target, X86Target};
use morello::tensorspec::TensorSpecAux;

#[inline(never)]
fn copy_actions_into_vec() {
    let rm2 = row_major(2);
    let logical_spec = LogicalSpec::Primitive::<X86Target>(
        PrimitiveBasics {
            typ: PrimitiveSpecType::Matmul { accum: false },
            spec_shape: smallvec![64, 64, 64],
            dtype: Dtype::Uint32,
        },
        PrimitiveAux(vec![
            TensorSpecAux {
                contig: rm2.contiguous_full(),
                aligned: true,
                level: CpuMemoryLevel::GL,
                layout: rm2,
                vector_size: None,
            };
            3
        ]),
        true,
    );
    black_box(logical_spec.actions().collect::<Vec<_>>());
}

main!(
    callgrind_args = "--simulate-wb=no", "--simulate-hwpref=yes",
        "--I1=32768,8,64", "--D1=32768,8,64", "--LL=8388608,16,64";
    functions = copy_actions_into_vec
);