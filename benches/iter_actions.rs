use iai_callgrind::{black_box, main};

use morello::common::Dtype;
use morello::layout::row_major;
use morello::lspec;
use morello::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType};
use morello::target::{CpuMemoryLevel::GL, X86Target};
use morello::tensorspec::TensorSpecAux;

#[inline(never)]
fn copy_actions_into_vec() {
    let rm2 = row_major(2);
    let logical_spec: LogicalSpec<X86Target> = lspec!(Matmul(
        [64, 64, 64],
        (u32, GL, rm2.clone()),
        (u32, GL, rm2.clone()),
        (u32, GL, rm2),
        serial
    ));
    black_box(logical_spec.actions().into_iter().collect::<Vec<_>>());
}

main!(
    callgrind_args = "--simulate-wb=no", "--simulate-hwpref=yes",
        "--I1=32768,8,64", "--D1=32768,8,64", "--LL=8388608,16,64";
    functions = copy_actions_into_vec
);
