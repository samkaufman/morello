# Introduction

Morello is a [synthesis][armandosynth]-based compiler of neural network forward passes,
primarily for CPUs.
It provides programmers with a [Halide][halide]-like language for manual optimization,
but it can also automatically synthesize implementations which minimize a cost model
(typically: maximizing throughput).

This book introduces the key ideas behind Morello by walking through manual optimization
of matrix multiplication, then by synthesizing matrix multiplications automatically.

[armandosynth]: https://people.csail.mit.edu/asolar/SynthesisCourse/TOC.
[halide]: https://halide-lang.org

## Status

Morello is under active development. The language and compiler internals are regularly
changing, and we do not yet ship versioned releases.

If you're interested in a stable library, you should target a particular version from
the main branch. If using Morello as a library, specify the dependency in `Cargo.toml`
such that it fixes a SHA1 hash:

```toml
morello = { git = "https://github.com/samkaufman/morello.git", rev = "..." }
```
