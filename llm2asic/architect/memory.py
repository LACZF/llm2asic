# llm2asic/architect/memory.py
"""内存与带宽预算检查。对应设计文档 architect.md §5.3。"""

from __future__ import annotations


def budget_check(archdesc, config: dict, onchip_cap_bytes: int = 0) -> dict:
    """检查片上权重 ROM + KV 缓存是否满足目标器件容量。"""
    res = archdesc.resource_estimate
    weight_bits = 0
    for e in archdesc.engines:
        if e.get("engine_type") == "gemm_engine":
            # 引擎权重总量在 Quantizer 报告；此处用估算
            pass
    # 使用 archdesc 保存的 gemm_macs_total 粗估权重 ROM 位宽
    gemm_macs = res.get("gemm_macs_total", 0)

    kv = None
    for e in archdesc.engines:
        if e.get("engine_type") == "kv_memory":
            kv = e["params"]

    # 权重 ROM 位宽粗估（每 MAC 一个权重，8bit）
    weight_bytes = gemm_macs  # 近似（byte=1）
    kv_bytes = 0
    max_seq = config.get("max_seq_len", 8)
    if kv:
        heads = kv["heads"]; hd = kv["head_dim"]; layers = kv["num_layers"]
        kv_bytes = 2 * layers * heads * hd * max_seq

    total = weight_bytes + kv_bytes
    ok = onchip_cap_bytes <= 0 or total <= onchip_cap_bytes
    return {
        "weight_rom_bytes": weight_bytes,
        "kv_bytes": kv_bytes,
        "total_bytes": total,
        "within_budget": ok,
    }
