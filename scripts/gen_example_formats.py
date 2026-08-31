#!/usr/bin/env python3
"""从 llama_tiny 的 weights.npz + model.yaml 生成多格式 example 目录。

产出（与 llama_tiny 完全等价的权重，供格式加载测试）：
  examples/llama_tiny_safetensors/model.safetensors + config.json
  examples/llama_tiny_onnx/model.onnx                + config.json
  examples/llama_tiny_bin/model.bin                    + model.yaml + config.json
"""

from __future__ import annotations

import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "examples", "llama_tiny")
OUTS = {
    "safetensors": os.path.join(ROOT, "examples", "llama_tiny_safetensors"),
    "onnx": os.path.join(ROOT, "examples", "llama_tiny_onnx"),
    "bin": os.path.join(ROOT, "examples", "llama_tiny_bin"),
}


def _cfg_dict() -> dict:
    return {
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "vocab_size": 32,
        "max_position_embeddings": 8,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1.0e-5,
        "tie_word_embeddings": False,
        "_name_or_path": "llama_tiny",
    }


def main() -> None:
    with np.load(os.path.join(SRC, "weights.npz")) as d:
        weights = {k: np.asarray(v) for k, v in d.items()}
    config = _cfg_dict()

    # ---------- safetensors ----------
    out = OUTS["safetensors"]
    os.makedirs(out, exist_ok=True)
    from safetensors.numpy import save_file
    save_file(weights, os.path.join(out, "model.safetensors"))
    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # ---------- onnx ----------
    out = OUTS["onnx"]
    os.makedirs(out, exist_ok=True)
    import onnx
    from onnx import helper, TensorProto
    inits = []
    wmap = {}
    for idx, (name, arr) in enumerate(sorted(weights.items())):
        t = helper.make_tensor(
            name=name,
            data_type=TensorProto.FLOAT,
            dims=list(arr.shape),
            vals=arr.astype(np.float32).flatten().tolist(),
            raw=False,
        )
        inits.append(t)
        wmap[name] = t
    inp = helper.make_tensor_value_info("tokens", TensorProto.INT64, [1])
    outi = helper.make_tensor_value_info("logits", TensorProto.FLOAT,
                                          list(weights["lm_head.weight"].shape))
    # 一个占位 Identity 图，仅承载 initializer 权重；真实图由后续 onnx-parser 展开。
    id_node = helper.make_node("Identity", ["tokens"],
                               ["logits_pre"], name="placeholder")
    cast_node = helper.make_node("Cast", ["logits_pre"], ["logits"], to=TensorProto.FLOAT)
    graph = helper.make_graph(
        [id_node, cast_node], "llama_tiny",
        [inp], [outi], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, os.path.join(out, "model.onnx"))
    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # ---------- raw fp32 bin ----------
    out = OUTS["bin"]
    os.makedirs(out, exist_ok=True)
    names = sorted(weights.keys())
    flat = np.concatenate([w.astype(np.float32).flatten() for w in
                           (weights[n] for n in names)])
    flat.tofile(os.path.join(out, "model.bin"))
    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    # 供原始二进制读取的形状/配置
    with open(os.path.join(out, "model.yaml"), "w", encoding="utf-8") as f:
        cfg = {
            "name": "llama_tiny",
            "head_dim": 4, "hidden": 16, "max_seq_len": 8, "num_heads": 4,
            "num_layers": 2, "rope_theta": 10000.0, "norm_eps": 1.0e-5,
            "tied_embedding": False, "vocab_size": 32,
            "weights": {n: list(weights[n].shape) for n in names},
        }
        import yaml
        yaml.safe_dump(cfg, f, sort_keys=True)

    print("生成完成:")
    for k, v in OUTS.items():
        print(f"  {k}: {os.listdir(v)}")


if __name__ == "__main__":
    main()
