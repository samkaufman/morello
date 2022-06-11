import functools
import io
import sys
from typing import Callable, List, Union

import tabulate
import termcolor

from . import cost, impl, tensor, tensor_namer
from .system_config.state import current_system


def _build_table(
    op: impl.Impl,
    show_spec: bool,
    cost_dict,
    show_utilization: bool,
    show_scheduled: bool,
    table: List[List[str]],
    tensor_name_fn: Callable[[Union[tensor.Tensor, tensor.Tile]], str],
    indent_depth: int,
    color: bool,
) -> None:
    system = current_system()
    ds = " " * (indent_depth * 2)
    new_row = [f"{ds}{_env_str(op, tensor_name_fn, underscore_inner=True, fancy=True)}"]
    if show_spec:
        new_row.append(str(op.spec))
    if cost_dict:
        new_row.append(cost_dict[op][1])
    if show_utilization:
        new_row.extend([str(op.peak_memory[b]) for b in system.ordered_banks])
    if show_scheduled:
        new_row.append("yes" if op.is_scheduled else "no")
    table.append(new_row)

    if isinstance(op, impl.Pipeline):
        for stage in op.stages:
            _build_table(
                stage,
                show_spec,
                cost_dict,
                show_utilization,
                show_scheduled,
                table,
                tensor_name_fn,
                indent_depth + 1,
                color=color,
            )
    else:
        for child in op.children:
            if not isinstance(child, impl.Impl):
                continue
            should_indent = not isinstance(op, impl.MoveLet)
            _build_table(
                child,
                show_spec,
                cost_dict,
                show_utilization,
                show_scheduled,
                table,
                tensor_name_fn,
                indent_depth + 1 if should_indent else indent_depth,
                color=color,
            )

    # As the stack unwinds, insert a "store" line if this was a Move for an output
    if isinstance(op, impl.MoveLet) and op.is_store:
        store_row = [f"{ds}{_store_env_str(op, tensor_name_fn, fancy=True)}"]
        while len(store_row) < len(table[-1]):
            store_row.append("")
        table.append(store_row)


def _whitespace_preserving_tabulate(*args, **kwargs):
    original_preserve_whitespace = tabulate.PRESERVE_WHITESPACE
    tabulate.PRESERVE_WHITESPACE = True
    result = tabulate.tabulate(*args, **kwargs)
    tabulate.PRESERVE_WHITESPACE = original_preserve_whitespace
    return result


def _env_str(
    imp: impl.AppliedImpl,
    name_tensor_fn: Callable[[tensor.TensorLike], str],
    underscore_inner: bool = False,
    fancy: bool = False,
) -> str:
    if isinstance(imp, impl.Pipeline):
        keyword = "pipeline"
        if fancy:
            keyword = termcolor.colored(keyword, attrs=["bold"])

        introduced_strs = []
        for stage in imp.stages[:-1]:
            introduced_strs.append(f"{name_tensor_fn(stage.output)}: {stage.output}")
        return f"{keyword} ({', '.join(introduced_strs)})"
    elif isinstance(imp, (impl.Loop, impl.SlidingWindowLoop)):
        istr = ")"
        if not underscore_inner:
            istr = f", {_env_str(imp.inner, name_tensor_fn, fancy=fancy)})"

        left_strs, right_strs = [], []
        for it in sorted(imp.tiles, key=str):
            left_strs.append(
                _loop_operand_str(it, name_tensor_fn=name_tensor_fn, fancy=fancy)
            )
            # TODO: Fix the following
            right_strs.append(name_tensor_fn(imp.operands[it.source]))
        assert left_strs and right_strs

        keyword = "tile"
        if imp.parallel:
            keyword = "par " + keyword
        arrow = "<-"
        if fancy:
            keyword = termcolor.colored(keyword, attrs=["bold"])
            arrow = termcolor.colored(arrow, attrs=["bold"])

        left_concat = ", ".join(left_strs)
        right_concat = ", ".join(right_strs)
        return f"{keyword} ({left_concat}) {arrow} ({right_concat}{istr}"
    elif isinstance(imp, (impl.MoveLet, impl.PadTranspack)):
        keyword = "move"
        if isinstance(imp, impl.PadTranspack):
            keyword = "padtranspack"
        elif imp.prefetching:
            keyword += "[p]"

        arrow = "<-"
        if fancy:
            keyword = termcolor.colored(keyword, attrs=["bold"])
            arrow = termcolor.colored(arrow, attrs=["bold"])
        return (
            f"{keyword}[{imp.destination.bank}]"
            f" {name_tensor_fn(imp.destination)}"
            f" {arrow} {name_tensor_fn(imp.operands[imp.source_idx])}"
        )
    elif isinstance(imp, impl.AppliedImpl):
        operands_str = ", ".join(name_tensor_fn(o) for o in imp.operands)
        return f"{type(imp).__name__}({operands_str})"
    else:
        raise ValueError(f"Unsupported type: {type(imp)}")


def _store_env_str(
    op,
    name_tensor_fn: Callable[[tensor.TensorLike], str],
    fancy: bool = False,
):
    keyword = "store"
    arrow = "<-"
    if fancy:
        keyword = termcolor.colored(keyword, attrs=["bold"])
        arrow = termcolor.colored(arrow, attrs=["bold"])
    return (
        f"{keyword} {name_tensor_fn(op.operands[op.source_idx])}"
        f" {arrow} {name_tensor_fn(op.destination)}"
    )


def _loop_operand_str(
    t, *, name_tensor_fn: Callable[[tensor.TensorLike], str], fancy: bool
):
    if isinstance(t, tensor.ConvolutionImageTile):
        prefix = "conv"
        if fancy:
            prefix = termcolor.colored(prefix, attrs=["bold"])
        desc_part = prefix + " " + "×".join(str(s) for s in t.dim_sizes)
    elif isinstance(t, tensor.Tile):
        desc_part = "×".join(str(s) for s in t.dim_sizes)
    else:
        desc_part = str(t)
    return f"{name_tensor_fn(t)}: {desc_part}"


def pprint(
    op: impl.Impl,
    show_spec: bool = True,
    show_cost: bool = True,
    show_utilization: bool = True,
    show_scheduled: bool = False,
    color: bool = True,
    file=sys.stdout,
    holes_ok: bool = False,
):
    def _cfmt(text, **kwargs):
        if not color:
            return text
        return termcolor.colored(text, **kwargs)

    op = op.to_applied()

    cost_dict = None
    headers = [""]
    if show_spec:
        headers.append("spec")
    if show_cost:
        cost_dict = cost.detailed_analytical_cost(op, holes_ok=holes_ok)
        headers.append("cost")
    if show_utilization:
        headers.extend(current_system().ordered_banks)
    if show_scheduled:
        headers.append("scheduled")
    table = []
    namer = tensor_namer.TensorNamer(op, tensors_to_color=op.operands)
    _build_table(
        op,
        show_spec,
        cost_dict,
        show_utilization,
        show_scheduled,
        table,
        tensor_name_fn=functools.partial(namer.name, color=color),
        indent_depth=0,
        color=color,
    )

    # Print a simple, static header.
    print(_cfmt("Impl", attrs=["bold", "underline"]), file=file)

    # Print the inputs and output
    inputs_str = _cfmt("Inputs: ", attrs=["bold"])
    inputs_str += ", ".join(
        f"{namer.name(inp, color=color)}: {str(inp)}" for inp in op.inputs
    )
    print(inputs_str, file=file)
    print(
        f"{_cfmt('Output:', attrs=['bold'])} "
        f"{namer.name(op.output, color=color)}: {str(op.output)}",
        file=file,
    )

    # Print the table
    print(_whitespace_preserving_tabulate(table, headers=headers), file=file)

    # Print an epilogue
    print(
        _cfmt("Note: ", attrs=["bold"])
        + _cfmt("move*", attrs=["bold"])
        + " is a MoveLet scope where the introduced tensor is an output.",
        file=file,
    )


def pformat(*args, **kwargs) -> str:
    if "file" in kwargs:
        raise ValueError("`file` keyword argument is not supported by pformat")
    kwargs = dict(kwargs)
    with io.StringIO() as buf:
        kwargs["file"] = buf
        pprint(*args, **kwargs)
        return buf.getvalue()