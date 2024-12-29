# First Schedule

**TODO:** Address or hide canonicalization. (This is a footgun.)

## Install

To get our feet wet with Morello, we'll write a program which generates the
simple dot product implementation we described in the previous section.

First, initialize a fresh bin and add the crate to your Cargo dependencies:

```sh
cargo init --bin ./dotproduct
cd dotproduct
cargo add --git https://github.com/samkaufman/morello morello
```

## Define the Goal Spec

In `src/main.rs`, begin by constructing the dot product Spec from the last
section. We'll use the `lspec` macro. While, in the last section, our Spec only
described whether or not a matrix multiplication should accumulate into the
output and the size of its three, we'll now include information about the data
type, memory level, and layout of its input and output tensors, as well as a
flag indicating that this the implementation should run on a single thread
(`serial`).

```rust
{{#include ../example_01.rs:specdef_use}}

{{#include ../example_01.rs:specdef}}
```


Notice that the Spec is parameterized by `X86Target`. Morello *targets* are
types which define a set of target-specific instructions, a set of available
Spec rewrites, and basic cost and memory models. (**TODO:** What else?
**TODO:** Describe re-targeting.) As you might have guessed, this example will
target X86, though we don't yet use any X86-specific intrinsics.

## Schedule an Implementation

Next, we construct an implementation by applying *scheduling operators* to the
Spec. This is called ``scheduling.'' We'll apply three operators, corresponding
to the three rewrites described in the previous section:

1. `to_accum` to introduce an accumulator followed by a `MatmulAccum`,
2. `split(&[1, 1])` to introduce a loop over the *k* dimension, and
3. `select(CpuKernel::MultAdd)` to replacement the body with the C multiply-accumulate.

Pull in the `SchedulingSugar` trait, which extends Specs with these operators,
as well as `CpuKernel`.

```rust
{{#include ../example_01.rs:schedule_use}}
```

Then apply:

```rust
{{#include ../example_01.rs:schedule}}
```

With `emit`, we can print the resulting implementation to stdout.

```rust
{{#include ../example_01.rs:emit}}
```

This will print the source for a complete C executable. Inside the `kernel` function,
you'll find the dot product implementation:

```c
/* (Zero((1×1, u32, RF), serial), [64, 1024, 0, 0])(_) */                                                                                                                                                      
assert(false);  // missing imp(n002[_])                                                                                                                                                                        
for (int n003 = 0; n003 < 32; n003++) {                                                                                                                                                                        
n002[(0)] += n000[(n003)] * n001[(n003)];  /* MultAdd */
```

But what's this `assert(false)`? This won't compile at all!

## Sub-Scheduling the Zero

**TODO:** Fill in.
