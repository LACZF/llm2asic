# llm2asic/quantizer/pipeline.py
"""Quantizer 主流程：LLM-IR -> 量化 + ROM + 元数据 + 参考执行器。

对应设计文档 quantizer.md（Step A-E）与 pipeline 中 quantizer.run()。
"""

from __future__ import annotations

import json
import os

import numpy as np

from ..ir.serialize import dump_json, dump_ir
from .quantizer import quantize_graph
from .model import QuantizedModel
from .reorder import reorder_weight, reorder_scale
from .rom import (write_weight_rom, write_scale_rom, write_lut_rom,
                  write_embed_rom)
from .kv_plan import plan_kv
from ..rtl_backend.numeric import F, REQUANT_S, ACT_BITS

EF = 10
RR = 2 * F


WEIGHT_FROM_NODE = {
    # 由 node 名 -> engine key 的反查由 quantizer.quantize_graph 完成
}


def run(llm_ir, quant_cfg, out_dir: str) -> QuantizedModel:
    """量化 LLM-IR 并写出产物。返回 QuantizedModel。"""
    qdir = os.path.join(out_dir, "quantizer")
    romdir = os.path.join(qdir, "weights_rom")
    os.makedirs(romdir, exist_ok=True)
    for f in os.listdir(romdir):
        os.remove(os.path.join(romdir, f))

    quant = quantize_graph(llm_ir, quant_cfg.quant)
    model = QuantizedModel(
        engines=quant["engines"],
        wte_q=quant["wte_q"],
        gammas=quant["gammas"],
        luts=quant["luts"],
        config=quant["config"],
        bit_width=quant["bit_width"],
        group_size=quant["group_size"],
        simd=quant_cfg.arch.simd,
        pe=quant_cfg.arch.pe,
    )

    simd = model.simd
    metadata = {"config": model.config, "weights": {}, "luts": {}, "gammas": {},
                "fv": {"F": F, "REQUANT_S": REQUANT_S, "EF": EF, "RR": RR}}

    # ---- 线性层 ROM ----
    for key, qw in model.engines.items():
        words, word_bits, _simd, ww = reorder_weight(qw, simd=simd)
        wfile = f"{key}_weight.mem"
        sfile = f"{key}_scale.mem"
        write_weight_rom(words, word_bits, os.path.join(romdir, wfile))
        num = reorder_scale(qw)
        write_scale_rom(num, 24, os.path.join(romdir, sfile))
        qw.rom_file = wfile
        qw.scale_rom_file = sfile
        metadata["weights"][key] = {
            "rom_file": wfile, "scale_rom_file": sfile,
            "rom_depth": len(words), "rom_width": word_bits,
            "bit_width": ww, "group_size": qw.group_size,
            "c_out": qw.c_out, "c_in": qw.c_in,
            "weight_source": qw.row_rename,
        }

    # ---- 嵌入 ROM ----
    embfile = "wte_q.mem"
    write_embed_rom(model.wte_q, 24, os.path.join(romdir, embfile))
    metadata["weights"]["wte_q"] = {"rom_file": embfile, "bit_width": 24}

    # ---- 归一化 gamma（定点常量 ROM）----
    for k, v in model.gammas.items():
        safe = k.replace(".", "_").replace("-", "_")
        fname = f"{safe}.mem"
        write_lut_rom(np.asarray(v, dtype=np.int64).reshape(-1), 24,
                      os.path.join(romdir, fname))
        metadata["gammas"][k] = {"rom_file": fname, "bit_width": 24,
                                 "shape": list(v.shape)}

    # ---- LUT ROM ----
    for fname, arr in model.luts.files.items():
        bits = { "rsqrt.mem": 24, "sigmoid.mem": 24,
                 "silu.mem": 24, "exp_neg.mem": 24, "recip.mem": 24 }[fname]
        write_lut_rom(arr, bits, os.path.join(romdir, fname))
        metadata["luts"][fname] = {"bit_width": bits, "depth": int(len(arr))}

    # ---- KV plan ----
    kv = plan_kv(model.config)
    dump_json(kv, os.path.join(qdir, "kv_plan.json"))
    metadata["kv_plan"] = kv

    # ---- QLLM-IR + 元数据 + 报告 ----
    qir_path = os.path.join(qdir, "qllm_ir.json")
    dump_ir(llm_ir, qir_path, weights_dir=os.path.join(qdir, "weights"))
    dump_json(metadata, os.path.join(qdir, "quant_metadata.json"))
    dump_json({
        "bit_width": model.bit_width,
        "group_size": model.group_size,
        "simd": simd,
        "num_linear": len(model.engines),
        "weight_rom_bytes_total": sum(
            (m["rom_depth"] * (m["rom_width"] // 8)) for m in metadata["weights"].values()
            if "rom_depth" in m),
        "kv_plan": kv,
    }, os.path.join(qdir, "quant_report.json"))

    return model
