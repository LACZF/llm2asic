#!/usr/bin/env python3
"""生成 gpt2_tiny 微型 GPT-2 示例（canonical 权重 + 声明式 model.yaml）。

canonical 权重含融合 c_attn（parser 在构建时拆为 q/k/v），并带 bias
（GPT-2 LayerNorm/Linear 均有 bias）。同样产出 HF 风格 safetensors/onnx 变体
以覆盖 normalize 的 GPT-2 别名映射。
"""

from __future__ import annotations

import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "examples", "gpt2_tiny")
OUT_HF = os.path.join(ROOT, "examples", "gpt2_tiny_hf")

H, HEADS, HD, LYRS, SEQ, VOCAB, INNER = 8, 2, 4, 1, 4, 16, 16
rng = np.random.default_rng(0)


def _f(name, *shape):
    seed = int(abs(hash(name))) % (2**32)
    r = np.random.default_rng(seed)
    return (r.standard_normal(shape) * 0.02).astype(np.float32)


def canonical_cfg() -> dict:
    return {
        "name": "gpt2_tiny",
        "architecture": "gpt2",
        "hidden": H, "num_heads": HEADS, "head_dim": HD,
        "num_layers": LYRS, "max_seq_len": SEQ, "vocab_size": VOCAB,
        "n_inner": INNER, "norm_eps": 1e-5, "tied_embedding": False,
    }


def hf_cfg_dict() -> dict:
    return {
        "model_type": "gpt2", "_name_or_path": "gpt2_tiny",
        "n_embd": H, "n_head": HEADS, "n_layer": LYRS,
        "n_positions": SEQ, "vocab_size": VOCAB, "n_inner": INNER,
        "layer_norm_eps": 1e-5,
    }


def build_weights() -> dict:
    w = {}
    w["wte"] = _f("wte", VOCAB, H)
    w["wpe"] = _f("wpe", SEQ, H)
    for i in range(LYRS):
        w[f"layers.{i}.ln_1.weight"] = _f(f"w{i}", H); w[f"layers.{i}.ln_1.bias"] = _f(f"b{i}", H)
        w[f"layers.{i}.ln_2.weight"] = _f(f"w{i}", H); w[f"layers.{i}.ln_2.bias"] = _f(f"b{i}", H)
        w[f"layers.{i}.c_attn.weight"] = _f("cattn", 3 * H, H)
        w[f"layers.{i}.c_attn.bias"] = _f("cb", 3 * H)
        w[f"layers.{i}.c_attn_o.weight"] = _f("co", H, H); w[f"layers.{i}.c_attn_o.bias"] = _f("cob", H)
        w[f"layers.{i}.mlp_fc.weight"] = _f("mfc", INNER, H); w[f"layers.{i}.mlp_fc.bias"] = _f("mfcb", INNER)
        w[f"layers.{i}.mlp_proj.weight"] = _f("mpo", H, INNER); w[f"layers.{i}.mlp_proj.bias"] = _f("mpob", H)
    w["final_norm.weight"] = _f("fn", H); w["final_norm.bias"] = _f("fnb", H)
    w["lm_head.weight"] = _f("lmh", VOCAB, H)
    return w


def to_hf_names(w: dict) -> dict:
    """canonical -> HF GPT-2 命名（覆盖 normalize）。"""
    m = {
        "wte": "transformer.wte.weight",
        "wpe": "transformer.wpe.weight",
    }
    hf = {}
    for k, v in w.items():
        hf[m.get(k, k)] = v
    for i in range(LYRS):
        hf[f"transformer.h.{i}.ln_1.weight"] = w[f"layers.{i}.ln_1.weight"]
        hf[f"transformer.h.{i}.ln_1.bias"] = w[f"layers.{i}.ln_1.bias"]
        hf[f"transformer.h.{i}.ln_2.weight"] = w[f"layers.{i}.ln_2.weight"]
        hf[f"transformer.h.{i}.ln_2.bias"] = w[f"layers.{i}.ln_2.bias"]
        hf[f"transformer.h.{i}.attn.c_attn.weight"] = w[f"layers.{i}.c_attn.weight"]
        hf[f"transformer.h.{i}.attn.c_attn.bias"] = w[f"layers.{i}.c_attn.bias"]
        hf[f"transformer.h.{i}.attn.c_proj.weight"] = w[f"layers.{i}.c_attn_o.weight"]
        hf[f"transformer.h.{i}.attn.c_proj.bias"] = w[f"layers.{i}.c_attn_o.bias"]
        hf[f"transformer.h.{i}.mlp.c_fc.weight"] = w[f"layers.{i}.mlp_fc.weight"]
        hf[f"transformer.h.{i}.mlp.c_fc.bias"] = w[f"layers.{i}.mlp_fc.bias"]
        hf[f"transformer.h.{i}.mlp.c_proj.weight"] = w[f"layers.{i}.mlp_proj.weight"]
        hf[f"transformer.h.{i}.mlp.c_proj.bias"] = w[f"layers.{i}.mlp_proj.bias"]
    hf["transformer.ln_f.weight"] = w["final_norm.weight"]
    hf["transformer.ln_f.bias"] = w["final_norm.bias"]
    hf["lm_head.weight"] = w["lm_head.weight"]
    hf.pop("wte", None); hf.pop("wpe", None)
    for i in range(LYRS):
        for k in list(hf):
            if k.startswith("layers."):
                hf.pop(k, None)
    return hf


def main() -> None:
    w = build_weights()
    # ---- canonical spec (.yaml + .npz) ----
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, "weights.npz"), **w)
    import yaml
    with open(os.path.join(OUT, "model.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(canonical_cfg(), f, sort_keys=True)

    # ---- HF 风格 safetensors + onnx（覆盖 GPT-2 normalize）----
    hf_w = to_hf_names(w)
    os.makedirs(OUT_HF, exist_ok=True)
    with open(os.path.join(OUT_HF, "config.json"), "w", encoding="utf-8") as f:
        json.dump(hf_cfg_dict(), f, indent=2)
    from safetensors.numpy import save_file
    save_file(hf_w, os.path.join(OUT_HF, "model.safetensors"))

    import onnx
    from onnx import helper, TensorProto
    inits = [helper.make_tensor(name=k, data_type=TensorProto.FLOAT, dims=list(v.shape),
                                vals=v.astype(np.float32).flatten().tolist())
             for k, v in hf_w.items()]
    inp = helper.make_tensor_value_info("tokens", TensorProto.INT64, [1])
    outi = helper.make_tensor_value_info("logits", TensorProto.FLOAT,
                                          list(hf_w["lm_head.weight"].shape))
    idn = helper.make_node("Identity", ["tokens"], ["logits_pre"])
    cast = helper.make_node("Cast", ["logits_pre"], ["logits"], to=TensorProto.FLOAT)
    graph = helper.make_graph([idn, cast], "gpt2_tiny", [inp], [outi], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, os.path.join(OUT_HF, "model.onnx"))

    print("gpt2_tiny 生成:")
    print(" ", OUT, os.listdir(OUT))
    print(" ", OUT_HF, os.listdir(OUT_HF))


if __name__ == "__main__":
    main()
