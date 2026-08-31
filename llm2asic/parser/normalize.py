# llm2asic/parser/normalize.py
"""外部模型权重/配置归一化为内部 canonical 形态。

把 HF 系（LLaMA / Qwen / Mistral / GPT-2 等）开箱自带的权重名与 config.json，
映射为 Parser（`formula.parser.build_graph`）所期望的统一命名与配置字段。

Paser 期望的 canonical 权重名：
    wte
    layers.{i}.input_layernorm.weight
    layers.{i}.self_attn.{q,k,v}_proj.weight
    layers.{i}.post_attention_layernorm.weight
    layers.{i}.mlp.{gate,up,down}_proj.weight
    final_norm.weight
    lm_head.weight
"""

from __future__ import annotations

import re
from typing import Callable

import numpy as np

# 需要从 HF 权重名剥离的顶层容器前缀
_STRIP_PREFIXES = ("model.", "transformer.", "llama.", "mistral.", "qwen2.",
                   "qwen.", "gpt2.", "bert.")

# 已知的别名映射（正则: canonical 局部名）
_ALIASES: list[tuple[re.Pattern, Callable[[re.Match], str]]] = [
    # GPT-2: h.{i}.ln_1 / ln_2 / attn.c_attn / mlp.c_fc,c_proj
    (re.compile(r"h\.(\d+)\.ln_1\.weight"), lambda m: f"layers.{m[1]}.input_layernorm.weight"),
    (re.compile(r"h\.(\d+)\.ln_2\.weight"), lambda m: f"layers.{m[1]}.post_attention_layernorm.weight"),
    # GPT-2 三合一投影 c_attn；v1 仅支持单头 q/k/v，无法直接拆分，标记不支持
    (re.compile(r"h\.(\d+)\.attn\.c_attn\.weight"), lambda m: f"!!unsupported:gpt2:c_attn:{m[1]}"),
    (re.compile(r"h\.(\d+)\.attn\.c_proj\.weight"), lambda m: f"!!unsupported:gpt2:o_proj:{m[1]}"),
    (re.compile(r"h\.(\d+)\.mlp\.c_fc\.weight"), lambda m: f"!!unsupported:gpt2:c_fc:{m[1]}"),
    (re.compile(r"h\.(\d+)\.mlp\.c_proj\.weight"), lambda m: f"!!unsupported:gpt2:c_proj:{m[1]}"),
]


def strip_prefix(name: str) -> str:
    for p in _STRIP_PREFIXES:
        if name.startswith(p):
            return name[len(p):]
    return name


def _canonical_weight_name(raw: str) -> str:
    """HF 权重名 -> canonical 名。无法映射的返回 None（交由调用方决定丢/报错）。"""
    name = strip_prefix(raw)

    # 已知别名（GPT-2 等）
    for pat, repl in _ALIASES:
        m = pat.match(name)
        if m:
            out = repl(m)
            if out.startswith("!!unsupported"):
                return None
            return out

    # LLaMA/Qwen/Mistral 系 —— 已是接近 canonical 的命名，仅处理容器前缀差异
    # model.layers.N.self_attn.q_proj.weight -> layers.N.self_attn.q_proj.weight
    # model.embed_tokens.weight / wte -> wte
    if name == "embed_tokens.weight" or name == "wte" or name == "transformer.wte.weight":
        return "wte"
    if name == "lm_head.weight":
        return "lm_head.weight"
    if name == "final_norm.weight" or name == "norm.weight":   # final norm
        return "final_norm.weight"
    # 归一化层可能与权重混用 norm.weight；post/input 已带名字
    pat_layer = re.compile(
        r"layers\.(\d+)\.(input_layernorm|post_attention_layernorm)"
        r"(\.(?:weight|gamma))?$")
    m = pat_layer.match(name)
    if m:
        ln = "input_layernorm" if m[2] == "input_layernorm" else "post_attention_layernorm"
        return f"layers.{m[1]}.{ln}.weight"

    pat_proj = re.compile(
        r"layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj)\.(weight|bias)?$")
    m = pat_proj.match(name)
    if m:
        return f"layers.{m[1]}.self_attn.{m[2]}.weight"

    pat_mlp = re.compile(r"layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")
    m = pat_mlp.match(name)
    if m:
        return f"layers.{m[1]}.mlp.{m[2]}.weight"

    # GPT-2 / 其它绝对位置类 / 无法识别
    return None


def normalize_config(raw: dict) -> dict:
    """把 HF config.json 推导为 Parser 期望的 config 字段。"""
    def g(*keys, default=None):
        for k in keys:
            if k in raw and raw[k] is not None:
                return raw[k]
        return default

    hidden = int(g("hidden_size", "n_embd", "d_model", default=32))
    num_layers = int(g("num_hidden_layers", "n_layer", default=2))
    heads = int(g("num_attention_heads", "n_head", default=4))
    head_dim = int(g("head_dim", default=max(1, hidden // heads)))
    vocab = int(g("vocab_size", "n_vocab", default=64))
    max_seq = int(g("max_position_embeddings", "n_positions", default=16))
    theta = float(g("rope_theta", default=10000.0))
    eps = float(g("rms_norm_eps", "layer_norm_eps", default=1e-5))
    tied = bool(g("tie_word_embeddings", default=False))

    return {
        "name": str(g("_name_or_path", "model", default="model")),
        "vocab_size": vocab,
        "hidden": hidden,
        "num_layers": num_layers,
        "num_heads": heads,
        "head_dim": head_dim,
        "max_seq_len": max_seq,
        "rope_theta": theta,
        "norm_eps": eps,
        "tied_embedding": tied,
    }


def normalize(raw_weights: dict) -> dict:
    """把外部权重映射为 canonical，返回 {canonical_name: ndarray}。

    无法映射的权重名被忽略（如 attention.o_proj、position embeddings 等），
    不计入计算图。重复 canonical 名取最先出现者。
    """
    out: dict = {}
    for raw_name, arr in raw_weights.items():
        canon = _canonical_weight_name(str(raw_name))
        if canon is None:
            continue
        if canon not in out:
            out[canon] = np.asarray(arr, dtype=np.float32)
    return out


def infer_config(raw_config: dict | None) -> dict:
    return normalize_config(raw_config or {})
