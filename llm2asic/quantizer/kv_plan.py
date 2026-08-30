# llm2asic/quantizer/kv_plan.py
"""KV 缓存规划：容量预算与分区。对应设计文档 quantizer.md §5（Step D）。"""

from __future__ import annotations

import numpy as np


def plan_kv(config: dict, dtype_bytes: int = 1, onchip_cap: int = 0) -> dict:
    """计算 KV 缓存规模并给出片上/片外分配。"""
    layers = config["num_layers"]
    heads = config["num_heads"]
    hd = config["head_dim"]
    max_seq = config["max_seq_len"]
    per_layer = 2 * heads * hd * max_seq * dtype_bytes
    total = per_layer * layers

    if onchip_cap <= 0:
        onchip_layers = layers
        external_layers = 0
        onchip_bytes = total
        external_bytes = 0
    else:
        n = min(layers, int(onchip_cap // max(1, per_layer)))
        onchip_layers = n
        external_layers = layers - n
        onchip_bytes = n * per_layer
        external_bytes = external_layers * per_layer

    return {
        "per_layer_bytes": per_layer,
        "total_bytes": total,
        "onchip_layers": onchip_layers,
        "external_layers": external_layers,
        "onchip_bytes": onchip_bytes,
        "external_bytes": external_bytes,
        "dtype_bytes": dtype_bytes,
        "kv_per_position_bytes": 2 * heads * hd * dtype_bytes,
    }
