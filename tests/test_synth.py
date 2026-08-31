# tests/test_synth.py
"""综合相关测试：可配置 rsqrt LUT 旋钮 + Yosys ASIC/FPGA 综合脚本生成。"""

import numpy as np
import pytest

from llm2asic.config import CompileConfig, SynthConfig
from llm2asic.rtl_backend.numeric import gen_luts
from llm2asic.rtl_backend.synth import gen_synth_script


def test_rsqrt_lut_bits_knob_sizes():
    """rsqrt_lut_bits 决定 rsqrt 查找表尺寸（默认保持 2^20+1 历史行为）。"""
    assert gen_luts().rsqrt.shape[0] == 2**20 + 1      # 默认 = 历史曲线
    assert gen_luts(rsqrt_bits=10).rsqrt.shape[0] == 2**10 + 1
    assert gen_luts(rsqrt_bits=12).rsqrt.shape[0] == 2**12 + 1
    # 值域契约在小表下仍成立
    lut = gen_luts(rsqrt_bits=10)
    assert lut.rsqrt[0] == 0
    assert np.all(lut.rsqrt[1:] > 0)


def _make_cfg(out_dir="build_out", backend="fpga", liberty=""):
    cfg = CompileConfig(model_path="dummy.yaml")
    cfg.out_dir = out_dir
    cfg.synth.backend = backend
    cfg.synth.liberty = liberty
    cfg.synth.pdk = "sky130hd"
    return cfg



def _fake_rtl_contents():
    """在临时目录生成最小 RTL 文件集，供脚本生成器读取。"""
    import os
    import tempfile
    d = tempfile.mkdtemp(prefix="synth_test_")
    for f in ("llama_mini_accel.sv", "gemv_0.sv", "gemv_1.sv",
              "attn.sv", "rmsnorm.sv", "sim_tb.sv"):
        open(os.path.join(d, f), "w").close()
    return d


def test_synth_script_asic_uses_nomap_and_liberty():
    """ASIC 脚本必须含 read_liberty、memory -nomap（避免 rsqrt ROM 展开成
    ~100 万单元而 OOM）、abc -liberty 与 dfflibmap。"""
    import tempfile
    d = _fake_rtl_contents()
    lib = tempfile.mktemp(suffix=".lib")
    open(lib, "w").close()
    cfg = _make_cfg(backend="asic", liberty=lib)
    s = gen_synth_script(cfg, d)
    assert f"read_liberty -lib {lib}" in s
    assert "memory -nomap" in s
    assert f"abc -liberty {lib}" in s
    assert f"dfflibmap -liberty {lib}" in s
    assert "hierarchy -top llama_mini_accel" in s
    # sim_tb 不应被读入综合
    assert "sim_tb.sv" not in s


def test_synth_script_fpga_uses_synth_xilinx():
    """FPGA 脚本使用 synth_xilinx 且不引用 liberty。"""
    d = _fake_rtl_contents()
    cfg = _make_cfg(backend="fpga")
    s = gen_synth_script(cfg, d)
    assert "synth_xilinx" in s
    assert "liberty" not in s
    assert "memory -nomap" not in s
    assert "abc" not in s


def test_synth_script_asic_requires_liberty_when_not_discoverable():
    """ASIC 后端既无 liberty 也无可发现 PDK 路径时应报错。"""
    d = _fake_rtl_contents()
    cfg = _make_cfg(backend="asic", liberty="")
    cfg.synth.pdk = "__no_such_pdk__"
    with pytest.raises(ValueError):
        gen_synth_script(cfg, d)
