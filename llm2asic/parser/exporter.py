# llm2asic/parser/exporter.py
"""LLM-IR 落盘：graph JSON + 权重 bin + manifest + diagnostics。对应设计文档 parser.md §5。"""

from __future__ import annotations

import json
import os

from ..ir.graph import GraphIR
from ..ir.serialize import dump_ir, graph_to_dict, dump_json


def export(g: GraphIR, out_dir: str) -> dict:
    """把 LLM-IR 写出。返回 {阶段: 路径} 便于报告。"""
    parser_dir = os.path.join(out_dir, "parser")
    weights_dir = os.path.join(parser_dir, "weights")
    os.makedirs(parser_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    # 1) 图 JSON (权重外置到 weights/)
    ir_path = os.path.join(parser_dir, "llm_ir.json")
    dump_ir(g, ir_path, weights_dir=weights_dir)

    # 2) manifest.json
    manifest = {
        wname: {
            "shape": list(w.shape),
            "dtype": "fp32",
            "file": w.data_file or f"{wname}.bin",
        }
        for wname, w in g.weights.items()
    }
    dump_json(manifest, os.path.join(parser_dir, "manifest.json"))

    # 3) diagnostics.json
    diag = {
        "unsupported_ops": [],
        "warnings": [],
        "num_nodes": len(g.nodes),
    }
    dump_json(diag, os.path.join(parser_dir, "diagnostics.json"))

    return {"parser": parser_dir}


def dump_graph(g: GraphIR) -> dict:
    """返回图 dict（便于测试直接使用）。"""
    return graph_to_dict(g)
