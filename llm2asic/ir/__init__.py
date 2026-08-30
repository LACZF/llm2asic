# llm2asic.ir
"""统一中间表示（IR）包。

定义贯穿全流程的数据结构（LLM-IR / QLLM-IR / ArchDesc 的基础），
算子清单（OP SET）与图工具。
"""

from .ops import Op, OP_SET
from .graph import TensorDesc, WeightDesc, Node, GraphIR, QuantInfo
from .serialize import (
    graph_to_dict,
    graph_from_dict,
    dump_ir,
    load_ir,
    dump_json,
    load_json,
)
from .passes import (
    topological_order,
    adjacency,
    predecessors,
    successors,
    validate_graph,
    consumers_of,
)

__all__ = [
    "Op",
    "OP_SET",
    "TensorDesc",
    "WeightDesc",
    "Node",
    "GraphIR",
    "QuantInfo",
    "graph_to_dict",
    "graph_from_dict",
    "dump_ir",
    "load_ir",
    "dump_json",
    "load_json",
    "topological_order",
    "adjacency",
    "predecessors",
    "successors",
    "validate_graph",
    "consumers_of",
]
