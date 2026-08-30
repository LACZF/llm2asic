# llm2asic/quantizer/fuse.py
"""图优化与算子融合。对应设计文档 quantizer.md §2（Step A）。

V1：RMSNorm/LayerNorm 折叠默认关闭（`fold_rmsnorm: false`）——保持算术与 RTL 简单、
误差面小。接口保留以支持后续启用。
"""

from __future__ import annotations


def fold_rmsnorm(graph) -> bool:
    """尝试 RMSNorm 折叠。V1 直接返回 False（不折叠）。"""
    return False


def apply_fusions(graph, enabled: bool = False):
    """对图应用融合。默认关闭。"""
    if not enabled:
        return graph, {"fused": 0}
    return graph, {"fused": 0}
