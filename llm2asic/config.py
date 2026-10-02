# llm2asic/config.py
"""编译配置加载 / 校验。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

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
class HlsBambuConfig:
    """Bambu (PandA HLS) 调用配置。"""
    binary: str = "bambu"
    device_name: str = "xc7a100t-1csg324-VVD"
    clock_period: float = 5.0
    compiler: str = "I386_CLANG16"   # 本地 PandA 只构建了 I386_CLANG16 前端
    opt_level: str = "-O2"
    interface: str = "INFER"
    soft_float: bool = True          # 关掉会缺 functional unit（sqrtf/expf…）
    faithful_rounding: bool = True   # -DFAITHFULLY_ROUNDED
    link_libm: bool = True           # -lm
    experimental_setup: str = ""     # 注意：会自带 -O0，覆盖 opt_level
    evaluation: str = ""             # 如 PERIOD,AREA,REGISTERS,DSPS,BRAMS
    simulate: bool = False
    simulator: str = "VERILATOR"
    mem_stub: bool = True            # 补齐 .v 引用但缺失的 .mem（全零占位）
    # 把 Bambu 生成 Verilog 里的 initial/$readmemb 改写成常量 case ROM。
    # 关掉就只剩占位 .mem，综合器依然读不到外部初始化数据。
    synth_cleanup: bool = True
    timeout: int = 3600


@dataclass
class HlsConfig:
    """HLS（GraphIR -> C++/ONNX -> Bambu）配置。

    这条路径与 ``backend: verilog`` 的原生 RTL 流程完全独立，
    不做量化，直接在浮点 HLS 内核上编译。
    """
    backend: str = "native"          # native | hls4ml | onnx
    pos: int = 0                     # 编译期已知的位置索引（decode 第 pos 步）
    run_bambu: bool = True           # 是否把 C++ 继续交给 Bambu 综合
    verify: bool = True              # 用 g++ + numpy 参考做数值校验（仅 native）
    rel_tol: float = 1e-4
    precision: str = "float"         # float | double（仅 native C 内核）
    # hls4ml
    hls4ml_precision: str = "float"
    hls4ml_reuse_factor: int = 1
    hls4ml_io_type: str = "io_parallel"
    # native C 内核
    n_buffers: int = 8
    pipeline_ii: int = 0             # >0 -> `#pragma HLS PIPELINE II=N`；
                                     # 默认 0 = 不发（见 c_kernel.CKernelConfig）
    # RTL 校验：Bambu 出 Verilog 后用 Yosys 展开检查（不做工艺映射）
    yosys_check: bool = True
    yosys: str = "yosys"
    yosys_timeout: int = 3600
    bambu: HlsBambuConfig = field(default_factory=HlsBambuConfig)


@dataclass
class CompileConfig:
    model_path: str
    out_dir: str = "build_out"
    backend: str = "verilog"
    quant: QuantConfig = field(default_factory=QuantConfig)
    arch: ArchConfig = field(default_factory=ArchConfig)
    synth: SynthConfig = field(default_factory=SynthConfig)
    hls: HlsConfig = field(default_factory=HlsConfig)
    # Parser 相关
    input_seq_len: int = 8
    fp_dtype: str = "fp32"
    # Quantizer 相关
    n_calib_tokens: int = 8
    # RTL / 验证
    max_cycles: int = 200000
    enable_sim: bool = True
    single_file: bool = False   # 额外产出单文件 RTL（所有模块合并成一个 .sv）
    act_bits: Optional[int] = None   # RTL 激活总线位宽；None=读模型 build.act_bits(默认16)
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
    hls_d = raw.get("hls", {}) or {}
    bambu_d = hls_d.get("bambu", {}) or {}
    hls = HlsConfig(
        backend=str(hls_d.get("backend", "native")),
        pos=int(hls_d.get("pos", 0)),
        run_bambu=bool(hls_d.get("run_bambu", True)),
        verify=bool(hls_d.get("verify", True)),
        rel_tol=float(hls_d.get("rel_tol", 1e-4)),
        precision=str(hls_d.get("precision", "float")),
        hls4ml_precision=str(hls_d.get("hls4ml_precision", "float")),
        hls4ml_reuse_factor=int(hls_d.get("hls4ml_reuse_factor", 1)),
        hls4ml_io_type=str(hls_d.get("hls4ml_io_type", "io_parallel")),
        n_buffers=int(hls_d.get("n_buffers", 8)),
        pipeline_ii=int(hls_d.get("pipeline_ii", 0)),
        yosys_check=bool(hls_d.get("yosys_check", True)),
        yosys=str(hls_d.get("yosys", "yosys")),
        yosys_timeout=int(hls_d.get("yosys_timeout", 3600)),
        bambu=HlsBambuConfig(
            binary=str(bambu_d.get("binary", "bambu")),
            device_name=str(bambu_d.get("device_name",
                                        HlsBambuConfig.device_name)),
            clock_period=float(bambu_d.get("clock_period", 5.0)),
            compiler=str(bambu_d.get("compiler", "I386_CLANG16")),
            opt_level=str(bambu_d.get("opt_level", "-O2")),
            interface=str(bambu_d.get("interface", "INFER")),
            soft_float=bool(bambu_d.get("soft_float", True)),
            faithful_rounding=bool(bambu_d.get("faithful_rounding", True)),
            link_libm=bool(bambu_d.get("link_libm", True)),
            experimental_setup=str(bambu_d.get("experimental_setup", "")),
            evaluation=str(bambu_d.get("evaluation", "")),
            simulate=bool(bambu_d.get("simulate", False)),
            simulator=str(bambu_d.get("simulator", "VERILATOR")),
            mem_stub=bool(bambu_d.get("mem_stub", True)),
            synth_cleanup=bool(bambu_d.get("synth_cleanup", True)),
            timeout=int(bambu_d.get("timeout", 3600)),
        ),
    )
    cfg = CompileConfig(
        model_path=model_path or str(build_d.get("model", "")),
        out_dir=out_dir or str(build_d.get("out_dir", "build_out")),
        backend=str(build_d.get("backend", "verilog")),
        quant=quant,
        arch=arch,
        synth=synth,
        hls=hls,
        input_seq_len=int(build_d.get("input_seq_len", 8)),
        fp_dtype=str(build_d.get("fp_dtype", "fp32")),
        n_calib_tokens=int(build_d.get("n_calib_tokens", 8)),
        max_cycles=int(build_d.get("max_cycles", 200000)),
        enable_sim=bool(build_d.get("enable_sim", True)),
        single_file=bool(build_d.get("single_file", False)),
        act_bits=int(build_d["act_bits"]) if "act_bits" in build_d else None,
        raw=raw,
    )
    return cfg
