# tests/test_formats.py
"""多格式模型加载：ONNX / .safetensors / .bin 与 .npz 等价。

验证 loader 能把不同磁盘格式归一化为一致的 canonical 权重/配置，
且产出与默认 .npz 完全相同的黄金参考（build -> quantize -> IntModel）。
"""

import os

import numpy as np
import pytest

from llm2asic.config import CompileConfig
from llm2asic.parser.builder import run_from_path
from llm2asic.parser.normalize import infer_config, normalize
from llm2asic.quantizer.pipeline import run as quantizer_run
from llm2asic.rtl_backend.reference import IntModel

EX = os.path.join(os.path.dirname(__file__), "..", "examples")

FORMATS = {
    "npz": ("llama_tiny/model.yaml", "spec"),
    "safetensors": ("llama_tiny_safetensors/model.safetensors", "safetensors"),
    "onnx": ("llama_tiny_onnx/model.onnx", "onnx"),
    "bin": ("llama_tiny_bin/model.bin", "bin"),
}


def _gold(path: str):
    p = os.path.abspath(path)
    import tempfile
    out = tempfile.mkdtemp(prefix="fmt_")
    ir = run_from_path(p)
    qm = quantizer_run(ir, CompileConfig(model_path=p, out_dir=out), out)
    qw = dict(qm.engines)
    qw["wte_q"] = qm.wte_q
    for k, v in qm.gammas.items():
        qw[k] = v
    m = IntModel(qw, qm.luts, qm.config)
    vocab = qm.config["vocab_size"]
    seq = qm.config["max_seq_len"]
    return np.array([m.run_decode_step(int((3 * i + 7) % vocab), i)
                     for i in range(seq)], dtype=np.int64)


@pytest.mark.parametrize("fmt", ["npz", "safetensors", "onnx", "bin"])
def test_format_loads_and_builds(fmt):
    rel, _ = FORMATS[fmt]
    p = os.path.abspath(os.path.join(EX, rel))
    assert os.path.exists(p), f"缺少 example 文件: {p}"
    ir = run_from_path(p)                    # 解析 + 量化前构建
    assert ir.config["hidden"] == 16
    assert ir.config["num_layers"] == 2
    assert ir.config["vocab_size"] == 32


@pytest.mark.parametrize("fmt", ["safetensors", "onnx", "bin"])
def test_format_gold_equals_npz(fmt):
    rel, _ = FORMATS[fmt]
    g_fmt = _gold(os.path.join(EX, rel))
    g_npz = _gold(os.path.join(EX, FORMATS["npz"][0]))
    assert np.array_equal(g_fmt, g_npz), f"{fmt} 与 npz 黄金参考不一致"


def test_normalize_hf_prefix():
    import numpy as np
    raw = {
        "model.embed_tokens.weight": np.ones((32, 16)),
        "model.layers.0.self_attn.q_proj.weight": np.ones((16, 16)),
        "model.layers.0.self_attn.k_proj.weight": np.ones((16, 16)),
        "model.layers.0.self_attn.v_proj.weight": np.ones((16, 16)),
        "model.layers.0.input_layernorm.weight": np.ones((16,)),
        "model.layers.0.post_attention_layernorm.weight": np.ones((16,)),
        "model.layers.0.mlp.gate_proj.weight": np.ones((16, 16)),
        "model.layers.0.mlp.up_proj.weight": np.ones((16, 16)),
        "model.layers.0.mlp.down_proj.weight": np.ones((16, 16)),
        "model.norm.weight": np.ones((16,)),
        "lm_head.weight": np.ones((32, 16)),
        # 应被忽略的不参与计算权重
        "model.layers.0.self_attn.o_proj.weight": np.ones((16, 16)),
        "model.layers.0.self_attn.rotary_emb.inv_freq": np.ones((2,)),
    }
    n = normalize(raw)
    assert n["wte"].shape == (32, 16)
    assert n["final_norm.weight"].shape == (16,)
    assert n["layers.0.self_attn.q_proj.weight"].shape == (16, 16)
    assert "o_proj.weight" not in str(n)
    assert "inv_freq" not in str(n)


def test_normalize_gpt2_alias_and_config():
    cfg = infer_config({"n_embd": 32, "n_layer": 3, "n_head": 8, "n_vocab": 128,
                        "n_positions": 64, "layer_norm_eps": 1e-5})
    assert cfg["hidden"] == 32
    assert cfg["num_layers"] == 3
    assert cfg["num_heads"] == 8
    assert cfg["vocab_size"] == 128
    assert cfg["max_seq_len"] == 64
