# llm2asic/quantizer/model.py
"""量化后的模型容器（QuantizedModel）：RTL Backend 的唯一算术输入。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..rtl_backend.numeric import QWeight, LUTSet


@dataclass
class QuantizedModel:
    engines: dict = field(default_factory=dict)      # key -> QWeight（线性层）
    wte_q: np.ndarray = None                         # 嵌入定点整数 [vocab,hidden]
    wpe_q: np.ndarray = None                         # 位置嵌入定点整数 [npos,hidden]（GPT-2）
    gammas: dict = field(default_factory=dict)       # key -> 定点 gamma/beta
    luts: LUTSet = None                              # 定点查找表
    config: dict = field(default_factory=dict)       # 模型配置
    bit_width: int = 8
    group_size: int = 128
    simd: int = 8
    pe: int = 8

    def qweight(self, key: str) -> QWeight:
        return self.engines[key]
