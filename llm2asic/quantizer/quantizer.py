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


def map_gpt2_norm_key(node_name: str, suffix: str) -> str:
    """GPT-2 LayerNorm 键：l0.n1 -> layers.0.ln_1.{suffix}；final.n -> final_norm.{suffix}。"""
    if node_name.startswith("l"):
        layer = int(node_name.split(".")[0][1:])
        kind = "ln_1" if node_name.split(".")[1] == "n1" else "ln_2"
        return f"layers.{layer}.{kind}.{suffix}"
    if node_name == "final.n":
        return f"final_norm.{suffix}"
    return node_name


def quantize_graph(graph, quant_cfg):
    """对 LLM-IR 做权重量化。

    返回 dict：
        engines: {key -> QWeight}        线性层
        wte_q:   np.ndarray            嵌入定点整数
        wpe_q:   np.ndarray            位置嵌入定点整数（GPT-2）
        gammas:  {key -> np.ndarray}    归一化 gamma/beta 定点整数
        luts:    LUTSet
        config:  dict                  模型配置
    """
    bw = quant_cfg.default_weight.bit_width
    group = quant_cfg.default_weight.group_size
    luts = gen_luts(rsqrt_bits=getattr(quant_cfg, "rsqrt_lut_bits", 20))
    arch = str(graph.config.get("architecture", "llama")).lower()

    engines = {}
    gammas = {}
    wte_q = None
    wpe_q = None

    for node in graph.nodes:
        op = node.op_type
        if op == "linear":
            key = map_engine_key(node.name)
            wname = node.weight_names[0]
            w = graph.weights[wname]
            qw = quantize_weight(w.data, bw, group, name=key)
            qw.row_rename = wname
            if len(node.weight_names) > 1:          # GPT-2 含 bias
                bname = node.weight_names[1]
                if bname in graph.weights and graph.weights[bname].data is not None:
                    bias = graph.weights[bname].data
                    qw.bias_q = np.rint(np.asarray(bias, np.float64)
                                        * (2.0 ** F)).astype(np.int64)
            engines[key] = qw
        elif op == "embedding":
            wname = node.weight_names[0]
            w = graph.weights[wname]
            if node.attributes.get("is_position") and arch == "gpt2":
                wpe_q = np.rint(np.asarray(w.data, np.float64) * (2.0 ** F)).astype(np.int64)
            else:
                wte_q = np.rint(np.asarray(w.data, np.float64) * (2.0 ** F)).astype(np.int64)
        elif op == "rmsnorm":
            wname = node.weight_names[0]
            w = graph.weights[wname]
            key = map_norm_key(node.name)
            gammas[key] = np.rint(np.asarray(w.data, np.float64) * (2.0 ** F)).astype(np.int64)
        elif op == "layernorm":
            wname, bname = node.weight_names
            gw = graph.weights[wname]
            bw2 = graph.weights[bname]
            gk = map_gpt2_norm_key(node.name, "gamma")
            bk = map_gpt2_norm_key(node.name, "beta")
            gammas[gk] = np.rint(np.asarray(gw.data, np.float64) * (2.0 ** F)).astype(np.int64)
            gammas[bk] = np.rint(np.asarray(bw2.data, np.float64) * (2.0 ** F)).astype(np.int64)

    quant = {
        "engines": engines,
        "wte_q": wte_q,
        "wpe_q": wpe_q,
        "gammas": gammas,
        "luts": luts,
        "config": dict(graph.config),
        "bit_width": bw,
        "group_size": group,
    }
    return quant
