# llm2asic/ir/passes.py
"""图分析工具：拓扑排序 / 邻接 / 校验 / 使用方。对应设计文档 parser.md §4,§6。"""

from __future__ import annotations

from collections import defaultdict, deque

from .graph import GraphIR, Node
from .ops import is_supported_op


def adjacency(g: GraphIR) -> dict[str, set[str]]:
    """节点名 -> 下游直接消费者节点名集合（node 级邻接）。"""
    consumed_by: dict[str, list[str]] = defaultdict(list)
    for node in g.nodes:
        for t in node.inputs:
            consumed_by[t].append(node.name)
    adj: dict[str, set[str]] = defaultdict(set)
    for node in g.nodes:
        for t in node.outputs:
            adj[node.name].update(consumed_by.get(t, []))
    return {n.name: set(adj.get(n.name, ())) for n in g.nodes}


def predecessors(g: GraphIR, node: Node) -> list[Node]:
    """返回直接前驱节点（其某输出是 node 的输入）。"""
    inputs = set(node.inputs)
    preds: list[Node] = []
    for n in g.nodes:
        if n is node:
            continue
        if set(n.outputs) & inputs:
            preds.append(n)
    return preds


def successors(g: GraphIR, node: Node) -> list[Node]:
    """返回直接后继节点。"""
    outs = set(node.outputs)
    succs: list[Node] = []
    for n in g.nodes:
        if n is node:
            continue
        if outs & set(n.inputs):
            succs.append(n)
    return succs


def consumers_of(g: GraphIR, tensor: str) -> list[Node]:
    """返回某个 tensor 的所有消费节点。"""
    return [n for n in g.nodes if tensor in n.inputs]


def topological_order(g: GraphIR) -> list[Node]:
    """Kahn 拓扑排序（node 级 DAG）；有环则抛 ValueError。"""
    nodes = g.nodes

    # tensor -> producer node name
    producer: dict[str, str] = {}
    for n in nodes:
        for t in n.outputs:
            producer[t] = n.name

    # node 级前驱集合（producer 为输入 tensor 的节点）
    preds: dict[str, set[str]] = {}
    for n in nodes:
        preds[n.name] = {producer[t] for t in n.inputs if t in producer}

    indeg = {name: len(ps) for name, ps in preds.items()}
    # 反向邻接：consumer 是 pred 的后继
    children: dict[str, list[str]] = {}
    for name, ps in preds.items():
        for p in ps:
            children.setdefault(p, []).append(name)

    q = deque([name for name, d in indeg.items() if d == 0])
    order: list[Node] = []
    by_name = {n.name: n for n in nodes}
    while q:
        name = q.popleft()
        order.append(by_name[name])
        for c in children.get(name, []):
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)
    if len(order) != len(nodes):
        raise ValueError("计算图存在环（cyclic），无法拓扑排序")
    return order


def validate_graph(g: GraphIR) -> list[str]:
    """校验节点/权重/形状一致性，返回错误列表（空 = 通过）。"""
    errors: list[str] = []
    seen_names: set[str] = set()
    tensor_names: set[str] = set(g.tensors.keys())
    for n in g.nodes:
        if n.name in seen_names:
            errors.append(f"重复节点名: {n.name}")
        seen_names.add(n.name)
        if not is_supported_op(n.op_type):
            errors.append(f"节点 {n.name} 使用了非内部算子: {n.op_type}")
        for t in n.inputs:
            if t not in tensor_names and t not in g.weights:
                errors.append(f"节点 {n.name} 的输入张量 {t} 未定义")
        for t in n.outputs:
            if t not in tensor_names:
                errors.append(f"节点 {n.name} 的输出张量 {t} 未在 tensors 中声明")
        for w in n.weight_names:
            if w not in g.weights:
                errors.append(f"节点 {n.name} 引用了未定义权重 {w}")
    for name in g.outputs:
        if name not in tensor_names:
            errors.append(f"模型输出 {name} 未定义")
    try:
        topological_order(g)
    except ValueError as e:
        errors.append(str(e))
    return errors
