# Morello

Morello is a synthesis-based compiler which generates fast neural network pipelines and kernels for CPUs. It consumes a neural network specification ("Spec" for short) and
generates a C implementation of that specification.

If you're new, the best place to start is the [book](http://samk.name/morello/book).
The book introduces the key ideas behind Morello, walking through increasingly
sophisticated neural networks and features of Morello. It's appropriate for any
developer interested in making their neural network inference workload run more quickly.

Additionally, a report discussing the system and its underlying ideas in more
detail can be found at:

[https://doi.org/10.48550/arXiv.2505.01637](https://doi.org/10.48550/arXiv.2505.01637)

## Installation

You can start running Morello locally by `git clone`ing the project.

An alternative is to launch a GitHub Codespace. This repository has a Dev Container
configuration, so launching a Codespace will now connect you to an environment set up
for Morello development (Rust toolchain, Clang, etc.).

## Automatic Synthesis

Try synthesizing a 2x2x2 matrix multiplication by running, from the cloned source directory, the following command:

```bash
cargo r --release -- matmul 2
```

Run `cargo r --release -- --help` for a list of other predefined specifications.

### Synthesis Databases

Synthesizing larger sizes (e.g., 16x16x16) can take a long time (hours or even days). To
speed up subsequent executions, Morello saves optimization decisions to disk when given
`--db` flag. For example:

```bash
cargo r --release -- --db morello.db matmul 2
```

This stores the optimal implementation for a 2x2x2 matrix multiplication as well as its
dependencies, including the optimal 1x1x1 matrix multiplication, optimal kernels for
moving data from global memory to registers, and many others. The next time you run
Morello to compute a 2x2x2 matrix multiplication, it will be near-instantaneous, but
also, if synthesizing a 4x4x4 matrix multiplication or a pipeline of matrix
multiplications, you'll have a head-start by reusing that database.

## Manual Scheduling

While Morello is primarily intended as a synthesizer, its IR can also be a convenient
way of manually lowering a specification to C. An example of manually scheduling a
matrix multiplication is given in
[morello/examples/simple_matmul_x86.rs](morello/examples/simple_matmul_x86.rs). To run it:

```bash
cargo r --example simple_matmul_x86
```

## Logging

Morello logs additional information via the [log](https://docs.rs/log/latest/log/) crate. Consider setting `RUST_LOG=info` in your shell environment.
