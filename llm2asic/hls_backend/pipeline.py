# llm2asic/hls_backend/pipeline.py
"""统一的 HLS 流程编排：GraphIR -> (ONNX | hls4ml | C 内核) -> Bambu -> Verilog。

对上层（CLI / Makefile）只暴露一个入口 `build_hls()`，用 `HlsBackend` 选择路径：

- ``native``    : GraphIR -> 自研 C 内核 -> Bambu -> Verilog
- ``hls4ml``    : GraphIR -> hls4ml HLS C++（可选再交给 Bambu）
- ``onnx``      : GraphIR -> ONNX（供 hls4ml 前端 / 第三方工具使用）

设计要点：
- 缺少工具（bambu / hls4ml）时**不抛异常**，把原因放进 `errors`，便于在
  没有 HLS 工具链的机器上仍然跑测试与 CI。
- 生成的每一步产物都记录在 `HlsResult` 里，便于排查。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import List, Optional

from ..ir.graph import GraphIR
from .bambu import BambuConfig, BambuResult, run_bambu, run_yosys_check
from .c_kernel import CKernelConfig, CKernelResult, gen_c_kernel
from .hls4ml_gen import Hls4mlConfig, Hls4mlResult, build_hls4ml

__all__ = ["HlsBackend", "HlsConfig", "HlsResult", "build_hls", "export_onnx"]

BACKENDS = ("native", "hls4ml", "onnx")


class HlsBackend:
    NATIVE = "native"
    HLS4ML = "hls4ml"
    ONNX = "onnx"


@dataclass
class HlsConfig:
    """HLS 流程配置。"""
    backend: str = HlsBackend.NATIVE
    out_dir: str = "build"
    # 编译期已知的位置索引（decode 第 pos 步），ONNX/hls4ml 用它切位置嵌入
    pos: int = 0
    # native
    precision: str = "float"           # float | double
    # hls4ml
    hls4ml_precision: str = "float"
    hls4ml_reuse_factor: int = 1
    hls4ml_io_type: str = "io_parallel"
    # 是否把 HLS C++ 继续交给 Bambu
    run_bambu: bool = True
    bambu: BambuConfig = field(default_factory=BambuConfig)
    ckernel: CKernelConfig = field(default_factory=CKernelConfig)
    hls4ml: Hls4mlConfig = field(default_factory=Hls4mlConfig)
    # 仿真校验：编译 C 内核并与 numpy 参考比对
    verify: bool = True
    gxx: str = "g++"
    rel_tol: float = 1e-4
    # RTL 校验：Bambu 出 Verilog 后用 Yosys 做展开检查（不映射工艺）
    yosys_check: bool = True
    yosys: str = "yosys"
    yosys_timeout: int = 3600


@dataclass
class HlsResult:
    """HLS 流程结果。"""
    ok: bool = False
    backend: str = ""
    out_dir: str = ""
    # 各阶段产物
    onnx_path: str = ""
    c_kernel_path: str = ""
    c_top_fname: str = ""
    hls4ml_project: str = ""
    hls4ml_cpp: str = ""
    hls4ml_layer_count: int = 0
    hls4ml_precision: str = ""
    verilog: List[str] = field(default_factory=list)
    top_verilog: str = ""
    mem_files: List[str] = field(default_factory=list)
    top_module: str = ""
    # 指标
    area: Optional[float] = None
    achieved_clock_ns: Optional[float] = None
    target_clock_ns: Optional[float] = None
    registers: Optional[int] = None
    dsps: Optional[int] = None
    brams: Optional[int] = None
    cycles: Optional[int] = None
    # RTL 校验
    yosys_ok: bool = False
    yosys_log: str = ""
    yosys_detail: str = ""
    # 校验
    verified: bool = False
    verify_detail: str = ""
    # 诊断
    bambu_version: str = ""
    cmd: List[str] = field(default_factory=list)
    report_path: str = ""
    log_path: str = ""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """人类可读的单行摘要。"""
        bits = [f"backend={self.backend}", f"ok={self.ok}"]
        if self.c_kernel_path:
            bits.append(f"c_kernel={os.path.basename(self.c_kernel_path)}")
        if self.hls4ml_cpp:
            bits.append(f"hls4ml_layers={self.hls4ml_layer_count}")
        if self.onnx_path:
            bits.append(f"onnx={os.path.basename(self.onnx_path)}")
        if self.verilog:
            bits.append(f"verilog={os.path.basename(self.top_verilog or self.verilog[0])}")
        if self.area is not None:
            bits.append(f"area={self.area:g}")
        if self.achieved_clock_ns:
            bits.append(f"clk={self.achieved_clock_ns:g}ns")
        if self.registers is not None:
            bits.append(f"ff={self.registers}")
        if self.cycles is not None:
            bits.append(f"cycles={self.cycles}")
        if self.yosys_ok:
            bits.append("yosys=True")
        if self.verified:
            bits.append("verified=True")
        return " ".join(bits)


def export_onnx(ir: GraphIR, out_dir: str, pos: int = 0, name: str = None):
    """GraphIR -> ONNX（薄封装，便于 CLI 单独使用）。"""
    from .onnx_export import export_onnx as _export

    return _export(ir, out_dir, pos=pos, name=name)


def _verify_c_kernel(ck: CKernelResult, ir: GraphIR, res: HlsResult,
                     gxx: str, rel_tol: float) -> None:
    """用 g++ 编译生成的 C 内核并与 numpy 浮点参考比对。

    这是不依赖 HLS 工具链也能做的最强校验：证明生成的 C 内核数值正确。
    """
    import shutil
    import subprocess
    import tempfile

    import numpy as np

    from .float_ref import ref_decode_step

    if shutil.which(gxx) is None:
        res.warnings.append(f"未找到 {gxx}，跳过 C 内核数值校验")
        return

    with tempfile.TemporaryDirectory() as td:
        exe = os.path.join(td, "kern")
        cmd = [gxx, "-std=c++14", "-O2", ck.top_path, "-o", exe, "-lm"]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if p.returncode != 0:
            res.errors.append(f"C 内核编译失败: {p.stderr.strip()[:400]}")
            return

        worst = 0.0
        for tok, pos in [(0, 0), (1, 1), (2, 2)]:
            run = subprocess.run([exe, str(tok), str(pos)],
                                 capture_output=True, text=True, timeout=300)
            if run.returncode != 0:
                res.errors.append(f"C 内核执行失败: {run.stderr.strip()[:200]}")
                return
            got = np.array([float(v) for v in run.stdout.split()],
                           dtype=np.float64)
            want = ref_decode_step(ir, tok, pos)
            if got.shape != want.shape:
                res.errors.append(
                    f"C 内核输出维度 {got.shape} 与参考 {want.shape} 不符")
                return
            scale = max(1e-12, float(np.abs(want).max()))
            worst = max(worst, float(np.abs(got - want).max()) / scale)

    if worst <= rel_tol:
        res.verified = True
        res.verify_detail = f"float 参考比对最大相对误差 {worst:.2e} <= {rel_tol:g}"
    else:
        res.errors.append(
            f"C 内核与 float 参考偏差过大: 相对误差 {worst:.2e} > {rel_tol:g}")


def build_hls(ir: GraphIR, cfg: Optional[HlsConfig] = None) -> HlsResult:
    """按 `cfg.backend` 执行 HLS 流程。"""
    cfg = cfg or HlsConfig()
    res = HlsResult(backend=cfg.backend, out_dir=cfg.out_dir)
    if cfg.backend not in BACKENDS:
        res.errors.append(
            f"未知后端 {cfg.backend!r}，可选: {', '.join(BACKENDS)}")
        return res

    os.makedirs(cfg.out_dir, exist_ok=True)
    name = ir.name or "llm2asic"

    # ---------------- ONNX ----------------
    if cfg.backend == HlsBackend.ONNX:
        try:
            ores = export_onnx(ir, cfg.out_dir, pos=cfg.pos, name=name)
        except Exception as e:                      # noqa: BLE001
            res.errors.append(f"ONNX 导出失败: {e}")
            return res
        res.onnx_path = getattr(ores, "path", "") or ""
        res.warnings.extend(getattr(ores, "unsupported", []) or [])
        res.ok = bool(res.onnx_path)
        if not res.ok:
            res.errors.append("ONNX 导出未产出 .onnx 文件")
        return res

    # ---------------- hls4ml ----------------
    if cfg.backend == HlsBackend.HLS4ML:
        # 用副本：派生值（project_name/out_dir）不能写回调用方复用的 config
        base = cfg.hls4ml or Hls4mlConfig()
        hcfg = replace(
            base,
            precision=cfg.hls4ml_precision,
            reuse_factor=cfg.hls4ml_reuse_factor,
            io_type=cfg.hls4ml_io_type,
            project_name=base.project_name or name,
        )
        try:
            hres: Hls4mlResult = build_hls4ml(ir, cfg.out_dir, hcfg, pos=cfg.pos)
        except Exception as e:                      # noqa: BLE001
            res.errors.append(f"hls4ml 生成失败: {e}")
            return res
        res.hls4ml_project = hres.project_dir
        res.hls4ml_cpp = hres.cpp_path
        res.hls4ml_layer_count = hres.layer_count
        res.hls4ml_precision = hcfg.precision
        res.errors.extend(hres.errors)
        res.warnings.extend(hres.warnings)
        if not hres.ok:
            return res
        if not cfg.run_bambu:
            res.ok = True
            res.top_module = hls4ml_top_name(hres)
            return res
        # Hls4mlConfig 没有 top_fname 字段，顶层名只能从 bridge 源码推断
        top = hls4ml_top_name(hres)
        return _bambu_from_cpp(res, hres.cpp_path, top, cfg)

    # ---------------- native C 内核 ----------------
    # 同样用副本：prefix/precision/emit_main 都是本次派生的
    ccfg = replace(
        cfg.ckernel or CKernelConfig(),
        precision=cfg.precision,
        emit_main=True,               # 供 g++ 校验与 Bambu 仿真
    )
    hls_dir = os.path.join(cfg.out_dir, "hls")
    try:
        ck = gen_c_kernel(ir, hls_dir, ccfg)
    except Exception as e:                      # noqa: BLE001
        res.errors.append(f"C 内核生成失败: {e}")
        return res
    res.c_kernel_path = ck.top_path
    res.c_top_fname = ck.top_fname
    res.top_module = ck.top_fname
    res.warnings.extend(ck.warnings)
    if not ck.ok:
        res.errors.extend(ck.errors)
        return res

    if cfg.verify:
        _verify_c_kernel(ck, ir, res, cfg.gxx, cfg.rel_tol)

    if not cfg.run_bambu:
        res.ok = bool(res.c_kernel_path) and not res.errors
        return res

    return _bambu_from_cpp(res, ck.top_path, ck.top_fname, cfg)


def hls4ml_top_name(hres: Hls4mlResult) -> str:
    """从 hls4ml 生成的 bridge 源码里推断顶层模块名。

    hls4ml 生成 `<project>_bridge.cpp`，其中暴露的顶层函数为
    `<project>(...)`；`firmware/<project>.cpp` 是模型实现。
    """
    import re as _re
    if hres.cpp_path and os.path.isfile(hres.cpp_path):
        base = os.path.basename(hres.cpp_path)
        m = _re.match(r"(.+)_bridge\.cpp$", base)
        if m:
            return m.group(1)
    return ""


def _bambu_from_cpp(res: HlsResult, cpp: str, top: str,
                    cfg: HlsConfig) -> HlsResult:
    """把一段 HLS C++ 交给 Bambu 生成 Verilog。"""
    bcfg = cfg.bambu or BambuConfig()
    from .bambu import bambu_version as _bv
    res.bambu_version = _bv(bcfg.binary)
    bres: BambuResult = run_bambu(cpp, os.path.join(cfg.out_dir, "bambu"),
                                  top, bcfg)
    res.verilog = list(bres.verilog)
    res.top_verilog = bres.top_verilog
    res.mem_files = list(bres.mem_files)
    res.area = bres.area
    res.achieved_clock_ns = bres.achieved_clock_ns
    res.target_clock_ns = bres.target_clock_ns
    res.registers = bres.registers
    res.dsps = bres.dsps
    res.brams = bres.brams
    res.cycles = bres.cycles
    res.cmd = list(bres.cmd)
    res.report_path = bres.report_path
    res.log_path = bres.log_path
    res.errors.extend(bres.errors)
    res.warnings.extend(bres.warnings)
    if bres.ok:
        res.top_module = bres.top_module or res.top_module
    res.ok = (not res.errors) and bool(res.verilog)

    if res.ok and cfg.yosys_check and res.top_verilog:
        ok, log, detail = run_yosys_check(
            res.top_verilog, res.top_module,
            os.path.dirname(res.top_verilog),
            timeout=cfg.yosys_timeout, yosys=cfg.yosys or None)
        res.yosys_ok = ok
        res.yosys_log = log
        res.yosys_detail = detail
        if not ok:
            res.errors.append(f"Yosys 展开失败: {detail}")
            res.ok = False
        else:
            res.warnings.append(detail)
    return res
