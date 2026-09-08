# llm2asic/config.py
"""编译配置加载 / 校验。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import yaml


@dataclass
class QuantScheme:
    """单个权重张量的量化方案。"""
    scheme: str = "symmetric_group"
    bit_width: int = 4
    group_size: int = 128
    symmetric: bool = True


@dataclass
class QuantConfig:
    default_weight: QuantScheme = field(default_factory=QuantScheme)
    exceptions: dict = field(default_factory=dict)     # 名字模式 -> QuantScheme
    activation_bit_width: int = 8
    threshold: float = 0.05     # 允许的最大数值退化
    sparse: bool = False
    rsqrt_lut_bits: int = 20    # rsqrt 查找表项数 = 2^bits（默认 2^20，ASIC 综合可调小）


@dataclass
class ArchConfig:
    pe: int = 16
    simd: int = 8
    pipeline_stages: int = 3
    target_device: str = "xczu7ev-ffvc1156-2-e"
    onchip_mem_bytes: int = 0     # 0 = 自动
    mode: str = "shared"
    clock_mhz: float = 250.0
    acc_width: int = 32


@dataclass
class SynthConfig:
    """综合配置（PDK 无关，支持 FPGA 或任意 ASIC 标准单元库）。"""
    backend: str = "fpga"          # "fpga" | "asic"
    family: str = "xc7"            # FPGA family（仅 backend=fpga，synth_xilinx）
    pdk: str = ""                  # PDK 名（仅 backend=asic，用于报告）
    liberty: str = ""              # ASIC 标准单元 .lib 路径（backend=asic 必需）
    clock_period_ns: float = 10.0  # 仅报告/后续 SDC，当前综合不使用时序约束


@dataclass
class CompileConfig:
    model_path: str
    out_dir: str = "build_out"
    backend: str = "verilog"
    quant: QuantConfig = field(default_factory=QuantConfig)
    arch: ArchConfig = field(default_factory=ArchConfig)
    synth: SynthConfig = field(default_factory=SynthConfig)
    # Parser 相关
    input_seq_len: int = 8
    fp_dtype: str = "fp32"
    # Quantizer 相关
    n_calib_tokens: int = 8
    # RTL / 验证
    max_cycles: int = 200000
    enable_sim: bool = True
    raw: dict = field(default_factory=dict)


def _parse_scheme(d: dict) -> QuantScheme:
    return QuantScheme(
        scheme=d.get("scheme", "symmetric_group"),
        bit_width=int(d.get("bit_width", 4)),
        group_size=int(d.get("group_size", 128)),
        symmetric=bool(d.get("symmetric", True)),
    )


def load_config(path: str, model_path: str = None, out_dir: str = None) -> CompileConfig:
    """从 YAML 加载编译配置。path 可为 None（使用默认）。"""
    raw: dict = {}
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    return parse_config(raw, model_path=model_path, out_dir=out_dir)


def parse_config(raw: dict, model_path: str = None, out_dir: str = None) -> CompileConfig:
    quant_d = raw.get("quant", {}) or {}
    qdefault = _parse_scheme(quant_d.get("default_weight", {})) if quant_d.get("default_weight") else QuantScheme()
    exceptions = {}
    for k, v in (quant_d.get("exceptions") or {}).items():
        exceptions[k] = _parse_scheme(v or {})
    quant = QuantConfig(
        default_weight=qdefault,
        exceptions=exceptions,
        activation_bit_width=int(quant_d.get("activation", {}).get("bit_width", 8)),
        threshold=float(quant_d.get("target_metric_deg", 0.05)),
        sparse=bool(quant_d.get("sparse", False)),
        rsqrt_lut_bits=int(quant_d.get("rsqrt_lut_bits", 14)),
    )
    arch_d = raw.get("arch", {}) or {}
    arch = ArchConfig(
        pe=int(arch_d.get("pe", 16)),
        simd=int(arch_d.get("simd", 8)),
        pipeline_stages=int(arch_d.get("pipeline_stages", 3)),
        target_device=str(arch_d.get("target_device", "xczu7ev-ffvc1156-2-e")),
        onchip_mem_bytes=int(arch_d.get("onchip_mem_bytes", 0)),
        mode=str(arch_d.get("mode", "shared")),
        clock_mhz=float(arch_d.get("clock_mhz", 250.0)),
        acc_width=int(arch_d.get("acc_width", 32)),
    )
    build_d = raw.get("build", {}) or {}
    synth_d = raw.get("synth", {}) or {}
    synth = SynthConfig(
        backend=str(synth_d.get("backend", "fpga")),
        family=str(synth_d.get("family", "xc7")),
        pdk=str(synth_d.get("pdk", "")),
        liberty=str(synth_d.get("liberty", "")),
        clock_period_ns=float(synth_d.get("clock_period_ns", 10.0)),
    )
    cfg = CompileConfig(
        model_path=model_path or str(build_d.get("model", "")),
        out_dir=out_dir or str(build_d.get("out_dir", "build_out")),
        backend=str(build_d.get("backend", "verilog")),
        quant=quant,
        arch=arch,
        synth=synth,
        input_seq_len=int(build_d.get("input_seq_len", 8)),
        fp_dtype=str(build_d.get("fp_dtype", "fp32")),
        n_calib_tokens=int(build_d.get("n_calib_tokens", 8)),
        max_cycles=int(build_d.get("max_cycles", 200000)),
        enable_sim=bool(build_d.get("enable_sim", True)),
        raw=raw,
    )
    return cfg
