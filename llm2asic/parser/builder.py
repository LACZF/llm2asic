# llm2asic/parser/builder.py
"""组装 GraphIR / 拓扑排序 / 死代码剪枝 / 校验。对应设计文档 parser.md §6。"""

from __future__ import annotations

from ..ir.graph import GraphIR
from ..ir.passes import validate_graph, topological_order
from .parser import build_graph, ModelParseError
from .loader import LoadedModel, load_model
from .shape import ShapeError


def prune(g: GraphIR) -> None:
    """删除不影响输出的死节点（上游保留）。"""
    outputs = set(g.outputs)
    needed = set(outputs)
    changed = True
    # 反向 BFS：从输出回推所有 producer
    producer_by_tensor: dict[str, set] = {}
    for n in g.nodes:
        for t in n.outputs:
            producer_by_tensor.setdefault(t, set()).add(n.name)
    # 需要保留的 tensor = 输出 + 被需要节点消费的输入
    needed_tensors = set(outputs)
    frontier = []
    # 找到生产 outputs 的节点
    for n in g.nodes:
        if set(n.outputs) & outputs:
            frontier.append(n)
    seen = set()
    while frontier:
        n = frontier.pop()
        if n.name in seen:
            continue
        seen.add(n.name)
        for t in n.inputs:
            needed_tensors.add(t)
            for p in producer_by_tensor.get(t, ()):
                pnode = g.find_node(p)
                if pnode:
                    frontier.append(pnode)
    g.nodes = [n for n in g.nodes if n.name in seen or set(n.outputs) & outputs]
    # 保留需要 tensor 对应的 producer 节点
    keep = set()
    for n in g.nodes:
        if set(n.outputs) & needed_tensors:
            keep.add(n.name)
    g.nodes = [n for n in g.nodes if n.name in keep]


def run(lm: LoadedModel) -> GraphIR:
    g = build_graph(lm)
    prune(g)
    errors = validate_graph(g)
    if errors:
        raise ModelParseError("LLM-IR 校验失败:\n" + "\n".join(errors))
    # 确保有合法拓扑序（会在 validate 时检查；这里再兜底）
    try:
        topological_order(g)
    except ValueError as e:
        raise ModelParseError(str(e)) from e
    return g


def run_from_path(model_path: str) -> GraphIR:
    lm = load_model(model_path)
    return run(lm)


def run_with_config(lm: LoadedModel, config=None) -> GraphIR:
    """保留 config 以便未来扩展（如固定 max_seq_len）。"""
    g = run(lm)
    if config is not None:
        g.config.update(config)
    return g
