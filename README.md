# Morello

Morello is an experimental scheduling language and auto-scheduler for tensor programs.

## Getting Started

Want to start playing with Morello? Great! You'll need Python 3, probably Python 3.9 or newer. The simplest way to get started is to install Morello's requirements with:

```sh
pip install -r requirements.txt
```

Want to isolate the Python interpreter and its dependencies? Use [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html)! Switch to a new Conda environment and run:

```sh
conda install --file requirements.txt
```

(You can achieve much the same thing with [venv](https://docs.python.org/3/library/venv.html).)

Once installed, try auto-scheduling a small matrix multiplication by running from the source directory root:
```sh
PYTHONPATH=. scripts/main.py matmul --cache seed_cache.pkl 8 4 8
```

If that works, you're in good shape. Find out how to schedule some other pre-baked Specs by running `scripts/main.py --help' or dig into the source code of [main.py](main.py) to schedule your own programs.

## Understanding the Output

The output of `main.py` should be the best found implementation of the given spec `Matmul((8×4), (4×8), (8×8))`. It should look something like this:

```
Schedule
Inputs: left: Tensor(8×4), right: Tensor(4×8)
Output: output: Tensor(8×8)
                                                                             spec                                              cost              lvl0    lvl1
---------------------------------------------------------------------------  ------------------------------------------------  ----------------  ------  ------
move[lvl=0] leftR <- left                                                    Matmul((8×4), (4×8), (8×8))                        3200 = 800 + _   128     0
move[lvl=0] rightR <- right                                                  Matmul((8×4, lvl=0), (4×8), (8×8))                 2400 = 800 + _   96      0
move*[lvl=0] outputR <- output                                               Matmul((8×4, lvl=0), (4×8, lvl=0), (8×8))          1600 = 1600 + _  64      0
tile (leftRt: 1×4, rightRt: 4×1, outputRt: 1×1) <- (leftR, rightR, outputR)  Matmul((8×4, lvl=0), (4×8, lvl=0), (8×8, lvl=0))      0 = 64 * _    0       0
  split (rightRtT: 1×1, leftRtT: 1×1) <- (rightRt, leftRt)                   Matmul((1×4, lvl=0), (4×1, lvl=0), (1×1, lvl=0))      0 = 4 * _     0       0
    Matmul(leftRtT, rightRtT, outputRt)                                      Matmul((1×1, lvl=0), (1×1, lvl=0), (1×1, lvl=0))      0             0       0
store output <- outputR
Note: move* is a MoveLet scope where the introduced tensor is an output.
```

The leftmost column contains the program implementation in Morello's Impl language.
Impl is defined recursively; every Impl program is an operation and zero or more nested subprograms.
Lines are indented to indicate nestedness with the exception of `move' instructions, for which nestedness is elided to improve readability.

Every Impl program, including each of its subprograms, has a Spec. The `spec` column contains the the Spec of the subprogram implemented by the Impl on that line.
For example, the above Matmul implementation is implemented by a move/load from the `left` parameter into a level-0 tensor named `leftR`, then a call into a continuation that implements the remainder of the program `Matmul((8×4, lvl=0), (4×8), (8×8))`. Notice that the subprogram Spec now expects a first operand in level 0 memory.

The `cost` column shows the total cost of the program implemented at that line, including a brief description on the right describing its calculation. For instance, the first line has a cost of 3200 and describes a cost of 800 (the cost of the move on that line) added to the cost of the subprogram.

The `lvl0` and `lvl1` columns show that line's Impl's peak memory consumed at levels 0 and 1 of the memory hierarchy.

### Tensor Lifetimes

Two Impls introduce new tensors: `move` and `pipeline`.

A `move' describes a load of a tensor from slow to faster memory and/or a store of a tensor from fast to slow memory. Whether a move corresponds to a load or a store is determined by the tensor's parameter site. The introduced tensor is live for the duration of the nested subprogram.

Store-only moves will be written with an asterisk (`move*`), and a `store` line will be introduced after the move's nested subprogram.
Note that `store` is not itself a real Impl; it is introduced only for readability.
As a conseqeuence, its Spec, cost, and peak memory measurement columns are blank.

A pipeline introduces tensors which bridge the output of each of its children to an input of the subsequent child.
These tensors live from the beginning of the producer's execution to the completion of the consumer. 
Pipelines are flattened when nested, so a nested pipeline will never appear in output.

Tiles over tensors are added by the tiling iterators `tile` and `convTile` and are live for the duration of the iterator.

## Scheduling Language

Implementations are scheduled using a tree of operators deriving from `Schedule`, and are defined in [ops.py](ops.py). They are:

  * `MoveLet` represents a memory move. It moves from slow to fast memory, and, in fast memory, may optionally change the layout of the underlying matrix.
  * `MatmulTilesLoop` represents a loop over the operand tiles required to calculate a tiled output.
  * `MatmulSplitLoop` represents a loop over the **k** dimension.
  * `Matmul`.
  * `DirectConv`.

Tree leaves are either `Matmul` or `DirectConv`.

Every `Schedule` has two operands: `lhs` and `rhs`. All except `Matmul` and `DirectConv` have an inner `Schedule`.

Operands are either tensors or views into tensors (tiles). These are defined in [tensor.py](tensor.py).

Schedules should be interpreted as having loop orders zipped with the logical dimensions of underlying tensors/tiles. The memory layout of the underlying buffers is described by the `layout` member on tensors/tiles.

### Scheduling Operators

Schedules are just Python [dataclasses](https://docs.python.org/3/library/dataclasses.html) and can be constructed manually, but it's much easier to chain applications of scheduling operators, starting with a `Matmul` or `DirectConv` instance. For example:

```python
spec = Matmul(Tensor(m, k, name="lhs"), Tensor(k, n, name="rhs"))
sched = spec.simple_tile(2, 2).complete()
op_pprint.pprint(sched)
```

Or: 

```python
spec = DirectConv(Tensor(m, k, name="lhs"), Tensor(k, n, name="rhs"))
sched = spec.simple_tile(2, 2).complete()
op_pprint.pprint(sched)
```

The operators are:

  * `tile` to split the Matmul into a loop over output tiles.
  * `split` to split the Matmul into a loop over the **k** dimension.
  * `move_left` and `move_right` to move the left or right operands to another level, and optionally change the layout.
  * `complete` to naively schedule the innermost `Matmul` down to 1x1 operands in registers.

## Cost Model

All costs are derived from moves, multiplied by the ranges of its containing loops. The model ignores the cost of stores. It schedules loads only. Tensor multiplication is free, but the cost model is only defined over matmuls. with 1x1, in-registers operands or direct convolutions with equally shaped image-filter pairs.

To determine the cost of a schedule, pass it to `analytical_cost` ([cost.py](cost.py)). The returned cost has no meaning other than being suitable for ranking the performance of schedules.

A proposed schedule is considered unschedulable—and `analytical cost` will raise `UnscheduledSpecError`—if it would move more data into a level of memory than is available according to our hypothetical system description in [system_config.py](system_config.py), if it fails to move operands into registers or tile to 1x1, or memory layout is changed at a level other than registers.

The crux of the cost model is implemented by `move_cost` in [cost.py](cost.py), which is pretty simple. The cost of a move is:

`10 * c * major * ceil(minor / line)`

where **major** is the rows dimension for a row-major source and the columns dimension otherwise, **minor** is the other, **line** is the cache line size in the toy system, and **c** is a coefficient associated with the cost of reading a cache line from the source's level of the memory hierarchy.
