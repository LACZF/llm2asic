# llm2asic/quantizer/quantizer.py
"""权重量化：把浮点权重量化为逐行定点，并生成所需的定点 LUT。

对应设计文档 quantizer.md §3（Step B）。产出数据由 reorder/rom 使用。
"""

from __future__ import annotations

import numpy as np

from ..rtl_backend.numeric import quantize_weight, QWeight, gen_luts, LUTSet, F


def map_engine_key(node_name: str) -> str:
    """把 GraphIR 节点名映射为引擎/权重键（RTL 引用名）。"""
    # 线性层：l0.q -> layers.0.q ; output_proj -> output_proj
    if node_name.startswith("l") and "." in node_name:
        layer = int(node_name[1:].split(".")[0])
        kind = node_name.split(".")[1]
        return f"layers.{layer}.{kind}"
    if node_name == "output_proj":
        return "output_proj"
    if node_name == "embed":
        return "wte"
    return node_name


def map_norm_key(node_name: str) -> str:
    if node_name.startswith("l"):
        layer = int(node_name.split(".")[0][1:])
        kind = "input_layernorm" if node_name.split(".")[1] == "n1" else "post_attention_layernorm"
        return f"layers.{layer}.{kind}.gamma"
    if node_name == "final.n":
        return "final_norm.gamma"
    return node_name


def quantize_graph(graph, quant_cfg):
    """对 LLM-IR 做权重量化。

    返回 dict：
        engines: {key -> QWeight}        线性层
        wte_q:   np.ndarray            嵌入定点整数
        gammas:  {key -> np.ndarray}    归一化 gamma 定点整数
        luts:    LUTSet
        config:  dict                  模型配置
    """
    bw = quant_cfg.default_weight.bit_width
    group = quant_cfg.default_weight.group_size
    luts = gen_luts()

    engines = {}
    gammas = {}
    wte_q = None

    for node in graph.nodes:
        op = node.op_type
        if op == "linear":
            key = map_engine_key(node.name)
            wname = node.weight_names[0]
            w = graph.weights[wname]
            qw = quantize_weight(w.data, bw, group, name=key)
            qw.row_rename = wname
            engines[key] = qw
        elif op == "embedding":
            wname = node.weight_names[0]
            w = graph.weights[wname]
            wte_q = np.rint(np.asarray(w.data, np.float64) * (2.0 ** F)).astype(np.int64)
        elif op == "rmsnorm":
            wname = node.weight_names[0]
            w = graph.weights[wname]
            key = map_norm_key(node.name)
            gammas[key] = np.rint(np.asarray(w.data, np.float64) * (2.0 ** F)).astype(np.int64)

    quant = {
        "engines": engines,
        "wte_q": wte_q,
        "gammas": gammas,
        "luts": luts,
        "config": dict(graph.config),
        "bit_width": bw,
        "group_size": group,
    }
    return quant
