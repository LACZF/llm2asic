# llm2asic/parser/shape.py
"""形状推导引擎（Shape Engine）。对应设计文档 parser.md §4。"""

from __future__ import annotations

from ..ir.graph import GraphIR, TensorDesc
from ..ir.passes import topological_order
from ..ir.ops import Op


class ShapeError(Exception):
    pass


def _vec(h: int, name: str = "") -> list:
    return [h]


def infer_shapes(g: GraphIR) -> None:
    """从已知 shapes 出发，按拓扑序传播输出形状。"""
    # 输入张量形状
    h = g.config.get("hidden", 32)
    g.tensors.setdefault("tokens", TensorDesc("tokens", [1], "int32"))
    g.tensors.setdefault("pos", TensorDesc("pos", [1], "int32"))

    try:
        order = topological_order(g)
    except ValueError as e:
        raise ShapeError(str(e))

    for node in order:
        # 已显式声明的输出形状保留
        outs = []
        for out_name in node.outputs:
            if out_name in g.tensors and g.tensors[out_name].shape:
                outs.append(list(g.tensors[out_name].shape))
            else:
                outs.append(None)

        shape = _out_shape(g, node)
        for i, out_name in enumerate(node.outputs):
            shp = outs[i] if outs[i] else shape.get(i, shape.get("all", None))
            if shp is None:
                raise ShapeError(f"无法推导节点 {node.name} 的输出形状")
            g.tensors[out_name] = TensorDesc(out_name, shp,
                                             _out_dtype(g, node))
    return


def _out_dtype(g: GraphIR, node) -> str:
    if node.op_type in (Op.EMBEDDING, Op.LINEAR, Op.RMSNORM, Op.LAYERNORM,
                        Op.ROPE, Op.ATTENTION, Op.ADD, Op.MUL, Op.SUB, Op.DIV,
                        Op.SILU, Op.GELU, Op.RELU, Op.SOFTMAX, Op.CONCAT,
                        Op.UNFLATTEN, Op.TRANSPOSE, Op.RESHAPE):
        return "fp32"
    return "int32"


def _out_shape(g: GraphIR, node) -> dict:
    op = node.op_type
    if op == Op.EMBEDDING:
        return {"all": _vec(g.config.get("hidden", 0))}
    if op == Op.LINEAR:
        of = node.attributes.get("out_features")
        # 读回 weight 形状确定输出维
        if node.weight_names:
            wname = node.weight_names[0]
            w = g.weights.get(wname)
            if w is not None and w.shape:
                of = w.shape[0]
        return {"all": _vec(of)}
    if op in (Op.RMSNORM, Op.LAYERNORM):
        # 形状与输入相同
        ns = node.attributes.get("normalized_shape") or []
        if ns:
            return {"all": list(ns)}
        return _same_as_input(g, node)
    if op == Op.ROPE:
        return _same_as_input(g, node)
    if op == Op.ATTENTION:
        return {"all": _vec(g.config.get("hidden", 0))}
    if op in (Op.ADD, Op.MUL, Op.SUB, Op.DIV, Op.SILU, Op.GELU, Op.RELU,
              Op.SOFTMAX, Op.CLONE, Op.COPY):
        return _same_as_input(g, node)
    if op == Op.CONCAT:
        # 沿非 axis 维与输入保持一致，axis 维求和
        a = g.tensors.get(node.inputs[0])
        b = g.tensors.get(node.inputs[1])
        if a and b:
            axis = int(node.attributes.get("axis", -1))
            shape = list(a.shape)
            shape[axis] = a.shape[axis] + b.shape[axis]
            return {"all": shape}
        return _same_as_input(g, node)
    if op == Op.TRANSPOSE:
        a = g.tensors.get(node.inputs[0])
        if a:
            dims = node.attributes.get("dims", [0, 1])
            shape = list(a.shape)
            d0, d1 = int(dims[0]), int(dims[1])
            shape[d0], shape[d1] = shape[d1], shape[d0]
            return {"all": shape}
        return _same_as_input(g, node)
    if op == Op.RESHAPE:
        return {"all": list(node.attributes.get("shape", []))}
    if op == Op.UNFLATTEN:
        return {"all": list(node.attributes.get("target_shape", []))}
    if op == Op.EMBEDDING and False:
        pass
    # 默认与输入相同
    return _same_as_input(g, node)


def _same_as_input(g: GraphIR, node) -> dict:
    inp = node.inputs[0] if node.inputs else None
    if inp and inp in g.tensors:
        return {"all": list(g.tensors[inp].shape)}
    return {}
