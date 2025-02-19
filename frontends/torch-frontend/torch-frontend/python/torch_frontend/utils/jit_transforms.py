import torch
import sys
import functools


def _nslice_slice_or_select_copy(graph, pattern):
    """
    slice + ... + slice/select + copy_
    """

    def is_valid():
        fill = pattern[-1]
        if fill.kind() != "aten::copy_":
            return False
        if pattern[-2].kind() not in ["aten::slice", "aten::select"]:
            return False
        for node in pattern[:-2]:
            if node.kind() != "aten::slice":
                return False
            return (
                node.inputsAt(2).toIValue() == 0
                and node.inputsAt(3).toIValue() == sys.maxsize
            )
        return True

    if not is_valid():
        return None, graph

    # availiable args.
    sc_input1 = pattern[0].inputsAt(0)  # self
    sc_input2 = pattern[-1].inputsAt(1)  # src
    dim = pattern[-2].inputsAt(1)
    start = pattern[-2].inputsAt(2)

    graph.setInsertPoint(pattern[-1])
    if pattern[-2].kind() == "aten::slice":
        # for slice + copy_
        end = pattern[-2].inputsAt(3)
        step = pattern[-2].inputsAt(4)
        slice_scatter = graph.create(
            "aten::slice_scatter",
            [sc_input1, sc_input2, dim, start, end, step],
            1,
        )
    else:
        # for select + copy_
        zero = graph.insertConstant(0)
        one = graph.insertConstant(1)
        end = graph.create("aten::add", [start, one], 1)
        end.output().setType(torch._C.IntType.get())
        unsqueeze = graph.create("aten::unsqueeze", [sc_input2, zero], 1)
        slice_scatter = graph.create(
            "aten::slice_scatter",
            [sc_input1, unsqueeze.output(), dim, start, end.output(), one],
            1,
        )
        graph.insertNode(end)
        graph.insertNode(unsqueeze)

    graph.insertNode(slice_scatter)

    graph.lint()

    return slice_scatter, graph


def _nslice_slice_or_select_fill(graph, pattern):
    """
    slice + ... + select/slice + fill_
    """

    def is_valid():
        fill = pattern[-1]
        if fill.kind() != "aten::fill_":
            return False
        if pattern[-2].kind() not in ["aten::slice", "aten::select"]:
            return False
        for node in pattern[:-2]:
            if node.kind() != "aten::slice":
                return False
            return (
                node.inputsAt(2).toIValue() == 0
                and node.inputsAt(3).toIValue() == sys.maxsize
            )
        return True

    if not is_valid():
        return None, graph

    graph.setInsertPoint(pattern[-1])
    sc_input1 = pattern[0].inputsAt(0)

    # build src.
    none = graph.insertConstant(None)
    zeros = graph.create(
        "aten::zeros_like", [sc_input1, none, none, none, none, none], 1
    )
    target_dim = pattern[-2].inputsAt(1)
    start = pattern[-2].inputsAt(2)
    one = graph.insertConstant(1)
    # for slice + fill_
    if pattern[-2].kind() == "aten::slice":
        end = pattern[-2].inputsAt(3)
        step = pattern[-2].inputsAt(4)
        sc_input2 = graph.create(
            "aten::slice", [zeros.output(), target_dim, start, end, step], 1
        )
        slice_scatter = graph.create(
            "aten::slice_scatter",
            [sc_input1, sc_input2.output(), target_dim, start, end, step],
            1,
        )
    else:
        # for select + fill
        end = graph.create("aten::add", [start, one], 1)
        end.output().setType(torch._C.IntType.get())
        sc_input2 = graph.create(
            "aten::slice", [zeros.output(), target_dim, start, end.output(), one], 1
        )
        slice_scatter = graph.create(
            "aten::slice_scatter",
            [sc_input1, sc_input2.output(), target_dim, start, end.output(), one],
            1,
        )
        graph.insertNode(end)

    graph.insertNode(zeros)
    graph.insertNode(sc_input2)
    graph.insertNode(slice_scatter)
    graph.lint()

    return slice_scatter, graph


def _select_slice_fill_(graph, pattern):
    """
    select + slice + fill_
    """

    def is_valid():
        if len(pattern) != 3:
            return False

        select = pattern[0]
        slice = pattern[1]
        fill = pattern[2]
        if (
            select.kind() != "aten::select"
            or slice.kind() != "aten::slice"
            or fill.kind() != "aten::fill_"
        ):
            return False
        start = slice.inputsAt(2).toIValue()
        end = slice.inputsAt(3).toIValue()
        if start != 0 or end != sys.maxsize:
            return False

        return True

    if not is_valid():
        return None, graph

    graph.setInsertPoint(pattern[-1])

    # rewrite select fill_ to slice_scatter.
    select = pattern[0]
    target_dim = select.inputsAt(1)
    start = select.inputsAt(2)
    one = graph.insertConstant(1)
    end = graph.create("aten::add", [start, one], 1)
    end.output().setType(torch._C.IntType.get())

    fill = pattern[2]
    sc_input1 = select.inputsAt(0)  # self
    none = graph.insertConstant(None)
    false = graph.insertConstant(False)
    value = fill.inputsAt(1)
    zeros = graph.create(
        "aten::zeros_like", [sc_input1, none, none, none, none, none], 1
    )
    fill_value = graph.create("aten::add", [zeros.output(), value, one], 1)
    sc_input2 = graph.create(
        "aten::slice", [fill_value.output(), target_dim, start, end.output(), one], 1
    )

    slice_scatter = graph.create(
        "aten::slice_scatter",
        [sc_input1, sc_input2.output(), target_dim, start, end.output(), one],
        1,
    )

    graph.insertNode(end)
    graph.insertNode(zeros)
    graph.insertNode(fill_value)
    graph.insertNode(sc_input2)
    graph.insertNode(slice_scatter)

    graph.lint()

    return slice_scatter, graph


copy_fill_rewrites = [
    _nslice_slice_or_select_copy,
    _nslice_slice_or_select_fill,
    _select_slice_fill_,
]


def replace_copy_fill_with_slice_scatter(graph):
    """
    - slice[(2, 0), (3, 9223372036854775807)]
    - any select/slice
    - copy_/fill_
    """

    def is_valid_node(node, level):
        if level == 0 or node.kind() in ["aten::slice", "aten::select"]:
            return True

        return False

    def dfs(node, level):
        pattern = []
        if is_valid_node(node, level):
            pattern.append(node)
        else:
            return pattern
        in_tensor = node.inputsAt(0)
        prev = in_tensor.node()
        pattern = dfs(prev, level + 1) + pattern
        return pattern

    def node_compare(lhs, rhs):
        if lhs[-1].isAfter(rhs[-1]):
            return 1
        elif lhs[-1].isBefore(rhs[-1]):
            return -1
        else:
            return 0

    # 1. find pattern points.
    valid_patterns = []
    for node in graph.nodes():
        if node.kind() not in ["aten::copy_", "aten::fill_"]:
            continue
        pattern = dfs(node, 0)
        if len(pattern) >= 2 and pattern[0].inputsAt(0).node().kind() not in [
            "aten::slice",
            "aten::select",
        ]:
            valid_patterns.append(pattern)

    def post_process(slice_scatter, graph):
        slice_scatter_use = None
        sc_input1 = slice_scatter.inputsAt(0)
        for use in sc_input1.uses():
            if use.user == slice_scatter:
                slice_scatter_use = use

        if slice_scatter_use:
            for use in sc_input1.uses():
                user = use.user
                if user == pattern[0] or not use.isAfter(slice_scatter_use):
                    continue

                for idx, value in enumerate(user.inputs()):
                    if value == sc_input1:
                        user.replaceInput(idx, slice_scatter.output())

        for old_node in reversed(pattern):
            old_node.destroy()
        graph.lint()

    # 2. do rewrite.
    sorted_patterns = sorted(valid_patterns, key=functools.cmp_to_key(node_compare))
    for pattern in sorted_patterns:
        for rewrite_func in copy_fill_rewrites:
            slice_scatter, graph = rewrite_func(graph, pattern)
            if slice_scatter is None:
                continue
            post_process(slice_scatter, graph)
            break

    return graph
