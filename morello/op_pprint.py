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
) -> None:
    system = current_system()
    ds = " " * (indent_depth * 2)
    new_row = [f"{ds}{op.env_str(tensor_name_fn, underscore_inner=True, fancy=True)}"]
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
            )

    # As the stack unwinds, insert a "store" line if this was a Move for an output
    if isinstance(op, impl.MoveLet) and op.is_store:
        store_row = [f"{ds}{op.store_env_str(tensor_name_fn, fancy=True)}"]
        while len(store_row) < len(table[-1]):
            store_row.append("")
        table.append(store_row)


def _whitespace_preserving_tabulate(*args, **kwargs):
    original_preserve_whitespace = tabulate.PRESERVE_WHITESPACE
    tabulate.PRESERVE_WHITESPACE = True
    result = tabulate.tabulate(*args, **kwargs)
    tabulate.PRESERVE_WHITESPACE = original_preserve_whitespace
    return result


def pprint(
    op: impl.Impl,
    show_spec: bool = True,
    show_cost: bool = True,
    show_utilization: bool = True,
    show_scheduled: bool = False,
    file=sys.stdout,
    holes_ok=False,
):
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
    namer = tensor_namer.TensorNamer(tensors_to_color=op.inputs + (op.output,))
    _build_table(
        op,
        show_spec,
        cost_dict,
        show_utilization,
        show_scheduled,
        table,
        tensor_name_fn=functools.partial(namer.name, color=True),
        indent_depth=0,
    )

    # Print a simple, static header.
    print(termcolor.colored("Impl", attrs=["bold", "underline"]), file=file)

    # Print the inputs and output
    inputs_str = termcolor.colored("Inputs: ", attrs=["bold"])
    inputs_str += ", ".join(
        f"{namer.name(inp, color=True)}: {str(inp)}" for inp in op.inputs
    )
    print(inputs_str, file=file)
    print(
        f"{termcolor.colored('Output:', attrs=['bold'])} "
        f"{namer.name(op.output, color=True)}: {str(op.output)}",
        file=file,
    )

    # Print the table
    print(_whitespace_preserving_tabulate(table, headers=headers), file=file)

    # Print an epilogue
    print(
        termcolor.colored("Note: ", attrs=["bold"])
        + termcolor.colored("move*", attrs=["bold"])
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
