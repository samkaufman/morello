# Morello

Morello is a synthesizer which generates fast neural network pipelines and kernels for
X86 and ARM CPUs. It consumes a neural network specification ("Spec" for short) and
generates a C implementation of that specification.

The easiest way to get started running Morello locally is to `git clone` the project
and, from the cloned source directory, synthesize one of the predefined specifications.
For example, to synthesize a 2x2x2 matrix multiplication:
```bash
cargo r --release -- matmul 2
```
(Run `cargo r --release -- --help` for a list of predefined specifications.)

A good alternative is to launch a GitHub Codespace. This repository has a Dev Container
configuration, so launching a Codespace will now connect you to an environment set up
for Morello development (Rust toolchain, Clang, etc.).

Synthesizing larger sizes (e.g., 16x16x16) can take a long time (hours or even days). To
speed up subsequent executions, Morello can memoize optimization decisions to disk when
given `--db` flag. For example:
```bash
cargo r --release -- --db morello.db matmul 2
```
This stores the optimal implementation for a 2x2x2 matrix multiplication as well as its
dependencies, including the optimal 1x1x1 matrix multiplication, optimal kernels for
moving data from global memory to registers, and many others. The next time you run
Morello to compute a 2x2x2 matrix multiplication, it will be near-instantaneous, but
also, if synthesizing a 4x4x4 matrix multiplication or a pipeline of matrix
multiplications, you'll have a head-start by reusing that database.

Additionally, if you're willing to store (both on disk and in memory) a larger
database, you can speed up synthesis by setting the environment variable
`MORELLO_USE_RLE_BLOCKS=1`.

## Logging

Morello logs useful, additional information via the [log](https://docs.rs/log/latest/log/) crate. Consider setting `RUST_LOG=info` in your shell environment to see these logs.

## Manual Scheduling

While Morello is primarily intended as a synthesizer, its IR can also be a convenient
way of manually lowering a specification to C. An example of manually scheduling a
matrix multiplication is given in
[examples/simple_matmul.rs](examples/simple_matmul.rs). To run it:
```bash
cargo r --example simple_matmul
```
