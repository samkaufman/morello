# Introduction

Morello is a compiler---and a library for building compilers---of neural
network inference procedures. It differs from most compilers in two ways.
First, it generates readable C code rather than LLVM IR or an executable, so
the output is usually passed to [Clang][clang] as a second step. Second, it is
[synthesis][armandosynth]-based; rather than generating code via a sequence of
optimization and lowering phases, it generates code which directly minimizes a
cost model.

Despite being primarily intended for synthesis, it also has first-class support
for manual programming via a [Halide][halide]-style scheduling language. Manual
programming and synthesis can be mixed.

This book will walk through the key ideas behind the Morello IR[^ir] and
manually scheduling a vector-matrix multiplication, then introduce automatic
synthesis of matrix multiplication and larger neural networks.


[armandosynth]: https://people.csail.mit.edu/asolar/SynthesisCourse/TOC.
[clang]: https://clang.llvm.org
[halide]: https://halide-lang.org
