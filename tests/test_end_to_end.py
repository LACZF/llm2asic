# tests/test_end_to_end.py
"""端到端：RTL 仿真与 numpy 黄金参考逐位一致。

需要环境中存在 Icarus Verilog（iverilog / vvp）。若不存在，仿真测试自动跳过，
其余（解析/量化/黄金参考）测试照常执行。
"""

import os
import shutil
import subprocess

import numpy as np
import pytest

from llm2asic.config import CompileConfig
from llm2asic.parser.builder import run_from_path
from llm2asic.quantizer.pipeline import run as quantizer_run
from llm2asic.rtl_backend import run as backend_run
from llm2asic.rtl_backend.reference import IntModel

MODEL = os.path.join(os.path.dirname(__file__), "..", "examples",
                     "llama_tiny", "model.yaml")
TOKENS = np.array([3, 7, 1, 15, 0, 5, 9, 2], dtype=np.int64)


def _have_iverilog() -> bool:
    return (shutil.which("iverilog") is not None
            and shutil.which("vvp") is not None)


@pytest.fixture(scope="module")
def qmodel(tmp_path_factory):
    mpath = os.path.abspath(MODEL)
    out = tmp_path_factory.mktemp("beam")
    cfg = CompileConfig(model_path=mpath, out_dir=str(out))
    ir = run_from_path(mpath)
    qm = quantizer_run(ir, cfg, str(out))
    return mpath, cfg, qm, str(out)


def _gold(qm, tokens):
    qw = dict(qm.engines)
    qw["wte_q"] = qm.wte_q
    for k, v in qm.gammas.items():
        qw[k] = v
    m = IntModel(qw, qm.luts, qm.config)
    return np.array([m.run_decode_step(int(tokens[i]), i)
                     for i in range(len(tokens))], dtype=np.int64)


def test_reference_rope_uses_sequence_position(qmodel):
    """黄金参考 RoPE 位置必须用序列下标 pos=i（与 RTL tokk 一致），
    而非 token 标识符本身 —— 这是逐位一致的前提。"""
    _, _, qm, _ = qmodel
    # pos=0 与 pos=1 的 RoPE 必须产生不同结果，且 KV 缓存跨 token 累积。
    qw = dict(qm.engines)
    qw["wte_q"] = qm.wte_q
    for k, v in qm.gammas.items():
        qw[k] = v
    m = IntModel(qw, qm.luts, qm.config)
    h = m.embed(int(TOKENS[0]))
    n1 = m.rmsnorm(h, qw["layers.0.input_layernorm.gamma"])
    r0 = m.rope(m.linear(n1, qw["layers.0.q"]), 0)
    r1 = m.rope(m.linear(n1, qw["layers.0.q"]), 1)
    assert not np.array_equal(r0, r1)


def test_reference_gold_shape_and_kv_accumulation(qmodel):
    _, _, qm, _ = qmodel
    g = _gold(qm, TOKENS)
    assert g.shape == (qm.config["max_seq_len"], qm.config["vocab_size"])
    # KV 缓存导致后一个 token 的 logits 与前一个不同（自回归）
    assert not np.array_equal(g[0], g[1])


@pytest.mark.skipif(not _have_iverilog(),
                    reason="需要 Icarus Verilog（iverilog/vvp）")
def test_rtl_sim_bit_exact(qmodel):
    mpath, cfg, qm, out = qmodel
    toks = TOKENS[: qm.config["max_seq_len"]]
    res = backend_run(mpath, cfg, tokens=toks)

    assert res.errors == []
    assert res.sim_ran
    assert res.bit_exact, \
        f"logits 未逐位一致: {res.logits_match}/{res.logits_total} " \
        f"worst={res.worst_abs_err}"


@pytest.mark.skipif(not shutil.which("iverilog"),
                    reason="需要 Icarus Verilog 进行编译验证")
def test_generated_rtl_compiles(qmodel):
    """生成的 RTL 应能被 iverilog 编译通过（即使不做完整仿真）。"""
    mpath, cfg, qm, out = qmodel
    toks = TOKENS[: qm.config["max_seq_len"]]
    backend_run(mpath, cfg, tokens=toks)
    rdir = os.path.join(out, "rtl")
    mod = f"{qm.config['name']}_accel"
    gemv_files = sorted(fn for fn in os.listdir(rdir)
                        if fn.startswith("gemv_") and fn.endswith(".sv"))
    cmd = ["iverilog", "-g2012", "-o", "/dev/null",
           os.path.join(rdir, f"{mod}.sv"),
           *[os.path.join(rdir, fn) for fn in gemv_files],
           os.path.join(rdir, "rmsnorm.sv"),
           os.path.join(rdir, "attn.sv"),
           os.path.join(rdir, "sim_tb.sv")]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"iverilog 失败:\n{proc.stdout}\n{proc.stderr}"
