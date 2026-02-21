# Specification Language

## Primitive Specs

Specs are applications of a function to tensor specifications as well as a flag
indicating whether or not the Spec is guaranteed to run on a single thread and limits
on the amount of data in each memory.

For example, the following constructs a Spec targeting X86:

```rust
Spec::<Avx2Target>(
    lspec!(Matmul(
        [M, K, N],
        (bf16, GL, row_major),
        (bf16, GL, col_major),
        (f32, GL, row_major),
        serial
    )),
    Avx2Target::max_mem(),
)
```

Tensor specifications describe each of the parameters of the function. Tensor
specifications describe the tensor's data type,
the memory where that tensor's data is stored,
the layout of the data,
whether that data is ``aligned'' (buffer address is some target-specific multiple),
a layout-specific description of how contiguous is the data, and,
in the data of data stored in a vector register file, the size of the vector register
(e.g., 128- or 256-bit on AVX2).

Available functions are listed as variants of the `PrimitiveSpecType` enum.

## Composition

**TODO:**: Fill in.
