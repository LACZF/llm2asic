# llm2asic/ir/graph.py
"""核心 IR 数据结构。对应设计文档 top.md §3.2。"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import numpy as np


@dataclass
class TensorDesc:
    """张量描述（形状 + 数据类型）。"""

    name: str
    shape: list[int]           # 全维度；标量 = []
    dtype: str = "fp32"        # fp32/bf16/fp16/int8/int4/int32 ...


@dataclass
class WeightDesc(TensorDesc):
    """权重描述。LLM-IR 下 data 为浮点；QLLM-IR 下 data 为量化整数。"""

    data: Optional[np.ndarray] = None       # 数值（不序列化，除非 small）
    scale: Any = None                       # float | np.ndarray | None
    zero_point: Any = None
    bit_width: Optional[int] = None
    group_size: Optional[int] = None
    data_file: Optional[str] = None         # 关联的 bin 文件（外置）
    quant_scheme: Optional[str] = None      # symmetric / symmetric_group ...


@dataclass
class Node:
    """计算图节点。"""

    name: str
    op_type: str                            # Op
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    attributes: dict = field(default_factory=dict)
    weight_names: list[str] = field(default_factory=list)
    quant: Optional[dict] = None            # 激活量化配置（QLLM-IR 填充）
    source: Optional[str] = None            # 上游算子名（溯源）


@dataclass
class QuantInfo:
    """单个权重的量化信息（对应 quant_metadata 中一项）。"""

    scheme: str = "symmetric_group"
    bit_width: int = 4
    group_size: int = 128
    scale_rom: Optional[str] = None
    scale_is_int: bool = True
    scale_mantissa_bits: int = 8
    scale_exponent_bits: int = 8
    rom_file: Optional[str] = None
    rom_depth: int = 0
    rom_width: int = 32
    layout: dict = field(default_factory=dict)
    requires: list[str] = field(default_factory=list)


@dataclass
class GraphIR:
    """容器：LLM-IR 与 QLLM-IR 共用同一结构。"""

    name: str = "unnamed"
    nodes: list[Node] = field(default_factory=list)
    tensors: dict[str, TensorDesc] = field(default_factory=dict)
    weights: dict[str, WeightDesc] = field(default_factory=dict)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    config: dict = field(default_factory=dict)
    quant_meta: dict = field(default_factory=dict)   # weight_name -> QuantInfo

    def tensor(self, name: str) -> TensorDesc:
        return self.tensors[name]

    def find_node(self, name: str) -> Optional[Node]:
        for n in self.nodes:
            if n.name == name:
                return n
        return None
