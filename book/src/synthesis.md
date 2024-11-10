# Automatic Synthesis

**TODO:** Write this chapter.

To synthesize, construct an in-memory database and call `synthesize` on a `Spec`:

```rust
let db = FilesDatabase::new(None, true, 1, 128, 1, None);
let implementation = spec.synthesize(&db, None);
```
