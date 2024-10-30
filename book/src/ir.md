# Hierarchical IR

Morello is built around a hierarchical IR[^ir] where the program and every
sub-program (i.e., program node) has a specification called a *Spec*. The key
idea is that programs can be built by iteratively rewriting goal Specs into
partial programs containing increasingly small and manageable Specs, finally
terminating in a complete C program when no nested Specs remain.

[^ir]: IR is short for *intermediate representation*, which is a program
    representation intended primarily for internal, automated manipulation by a
    compiler. This is in contrast to a programming language, which is a
    representation meant for reading and writing by a programmer.

## Schedule a Dot Product

As an example, consider a dot product of two 32-value vectors,
described by the (simplified) Spec `Matmul(1x32x1)`:

```c
//~#include <stdio.h>
//~
//~int main() {
//~  int output;
//~  int lhs[32], rhs[32];
output = 0;
for (unsigned i = 0; i < 32; i++) {
  output += lhs[i] * rhs[i];
}
//~  return 0;
//~}
```


With Morello, a programmer begins with a partial program containing only the
goal Spec:
```c
// Matmul(1x32x1)
```

The first step is to rewrite that trivially nested Spec into a partial program
which initializes an accumulator and contains a new, accumulating Spec.
```c
output = 0;
// MatmulAccum(1x32x1)
```

Next, that nested Spec is replaced by a loop followed by a version of the
original Spec tiled over the *k* dimension:
```c
output = 0;
for (unsigned i = 0; i < 32; i++) {
  // MatmulAccum(1x1x1)
}
```

Finally, that nested Spec is replaced with a C multiply-accumulate statement:

```c
output = 0;
for (unsigned i = 0; i < 32; i++) {
  output += lhs[i] * rhs[i];
}
```

At this point, no nested Specs remain and the program is complete.
