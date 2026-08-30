# llm2asic/architect/dse.py
"""设计空间探索（DSE）：并行度 × 位宽 -> 面积/吞吐权衡。对应设计文档 architect.md §6。"""

from __future__ import annotations


def estimate_resources(pe: int, simd: int, mode: str) -> dict:
    dsp = pe * simd
    lut = dsp * 64 + pe * 32
    return {"DSP": dsp, "LUT": lut, "BRAM": 4, "URAM": 0}


def estimate_throughput(qmodel, pe: int, simd: int, mode: str) -> float:
    """粗略吞吐（MAC/cycle）。decode GEMV：每行 C_in/simd 周期产出 PE 输出。"""
    hidden = qmodel.config["hidden"]
    total_macs = sum(e.c_out * e.c_in for e in qmodel.engines.values())
    cycles_per_pass = (hidden // simd) * (pe)
    return total_macs / max(1, cycles_per_pass)


def explore(qmodel, device_limits: dict = None):
    device_limits = device_limits or {"DSP": 4096, "LUT": 200000}
    candidates = []
    for pe in [8, 16, 32]:
        for simd in [4, 8, 16]:
            res = estimate_resources(pe, simd, "shared")
            if res["DSP"] <= device_limits["DSP"] and res["LUT"] <= device_limits["LUT"]:
                thru = estimate_throughput(qmodel, pe, simd, "shared")
                candidates.append({
                    "pe": pe, "simd": simd,
                    "throughput_per_area": thru / max(1, res["DSP"]),
                    "resources": res,
                })
    candidates.sort(key=lambda c: -c["throughput_per_area"])
    return candidates[:5]
