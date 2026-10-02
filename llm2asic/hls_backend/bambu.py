# llm2asic/hls_backend/bambu.py
"""Bambu (PandA HLS) 调用层：把 HLS C++ 内核综合成 Verilog。

Bambu 不是 pip 包，本模块只负责**发现**可执行文件、拼装命令行、收集产物与报告。
若 `bambu` 不在 PATH 上，会返回 `BambuResult(ok=False, errors=[...])` 而不是抛异常，
方便上层在没有 HLS 工具链的环境里继续跑测试。

命令行选项依据 PandA 官方文档：
  https://panda.deib.polimi.it/?page_id=916

对应设计文档 hls_backend.md §5。
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .verilog_synth import (
    make_rom_synthesizable,
    scan_simulation_only,
)

__all__ = [
    "BambuConfig", "BambuResult", "bambu_version", "bambu_available",
    "run_bambu", "parse_bambu_report", "detect_top_module", "run_yosys_check",
]

# PandA 官方示例常用的器件名；可用 --device-name 覆盖
DEFAULT_DEVICE = "xc7a100t-1csg324-VVD"

# 本地 PandA 构建（clang17 兼容补丁）唯一可用的前端
DEFAULT_COMPILER = "I386_CLANG16"

# Bambu 生成的 .v 里引用初始化文件的位置；用于收集/补齐 .mem
_MEM_REF_RE = re.compile(r'MEMORY_INIT_file\s*[=(]\s*"([^"]+)"')
# 测试平台/仿真中间产物目录，不应被当成设计 Verilog
_NOT_DESIGN_DIRS = {"beh_sim", "simulation", "verilator_obj", "HLS_output"}


@dataclass
class BambuConfig:
    """Bambu 调用配置。"""
    binary: str = "bambu"
    device_name: str = DEFAULT_DEVICE
    clock_period: float = 5.0        # ns
    top_fname: str = ""              # 顶层函数名（一般由调用方填）
    compiler: str = DEFAULT_COMPILER  # I386_CLANG16；留空交给 Bambu 自动选
    opt_level: str = "-O2"
    interface: str = "INFER"
    # 浮点：本地 softfloat 实现；不显式链接 libm 时 sqrtf/expf 等
    # 会因"没有对应的 functional unit"而在分配阶段失败。
    soft_float: bool = True
    faithful_rounding: bool = True   # -DFAITHFULLY_ROUNDED
    link_libm: bool = True           # -lm
    # 注意：BAMBU setup 会附带 -O0，会覆盖 opt_level，故默认留空。
    experimental_setup: str = ""     # 如 "BAMBU-PERFORMANCE"
    evaluation: str = ""             # 如 "PERIOD,AREA,CLOCK_SLACK,REGISTERS,DSPS,BRAMS"
    simulate: bool = False
    simulator: str = "VERILATOR"
    # 生成的 .v 引用了不存在的 .mem 时（如 --simulate 的 array.mem），
    # 是否补一个全零占位文件，好让下游 Yosys 能展开。
    mem_stub: bool = True
    # 把 initial/$readmemb 改写成可综合的常量 case ROM。Bambu 生成的
    # 存储体模板一定带这段，只在仿真有效；综合器不读外部 .mem，
    # ASIC 流程直接报错、FPGA 流程常静默丢初值。默认开。
    synth_cleanup: bool = True
    extra_args: List[str] = field(default_factory=list)
    timeout: int = 3600
    cwd_name: str = "bambu_run"      # 工作目录名（相对 out_dir）


@dataclass
class BambuResult:
    """Bambu 运行结果。"""
    ok: bool = False
    cmd: List[str] = field(default_factory=list)
    returncode: int = -1
    hls_output_dir: str = ""         # HLS_output/
    verilog: List[str] = field(default_factory=list)
    top_verilog: str = ""            # <top>.v（Bambu 写在工作目录，不是 HLS_output/）
    mem_files: List[str] = field(default_factory=list)
    mem_stubbed: List[str] = field(default_factory=list)
    # ROM 改写结果（initial/$readmemb -> 常量 case）
    rom_files_baked: List[str] = field(default_factory=list)
    rom_readmem_removed: int = 0
    unsynthesizable: Dict[str, int] = field(default_factory=dict)
    top_module: str = ""
    # 报告指标（未找到时为 None）
    area: Optional[float] = None
    achieved_clock_ns: Optional[float] = None
    target_clock_ns: Optional[float] = None
    registers: Optional[int] = None
    dsps: Optional[int] = None
    brams: Optional[int] = None
    cycles: Optional[int] = None
    report_path: str = ""
    log_path: str = ""
    stdout_tail: str = ""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def verilog_paths(self) -> List[str]:
        return self.verilog


def bambu_version(binary: str = "bambu") -> str:
    """返回 `bambu --version` 的首行；找不到或执行失败返回 ''。"""
    exe = binary or "bambu"
    if os.path.sep not in exe and shutil.which(exe) is None:
        return ""
    try:
        p = subprocess.run([exe, "--version"], capture_output=True,
                           text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return ""
    out = (p.stdout or p.stderr or "").strip()
    return out.splitlines()[0].strip() if out else ""


def bambu_available(binary: str = "bambu") -> bool:
    """`bambu` 是否可用。"""
    return bool(bambu_version(binary))


def _read(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def parse_bambu_report(hls_output_dir: str, log_text: str = "",
                       top: str = "") -> dict:
    """从 Bambu 产物里尽力提取面积/时序/资源/周期指标。

    关键事实：PandA 2024.10 **只把指标打到 stdout**，`HLS_output/` 下没有任何
    汇总报告（`bambu_results_0.xml` 里只有 CYCLES 和执行时间）。
    所以这里以调用方保存下来的日志文本为主源，`HLS_output/` 为补充。

    Bambu 不同版本措辞不同（`<top>.log`、`<top>.xml`、`bambu.log`…），
    逐个模式匹配，找不到就留空 —— 指标是"有则用之"，不应因格式变化让流程失败。
    """
    out: dict = {}

    # 0) bambu_results_0.xml：只有 CYCLES / HLS_execution_time
    if os.path.isdir(hls_output_dir):
        for name in sorted(os.listdir(hls_output_dir)):
            if not name.lower().endswith(".xml"):
                continue
            try:
                root = ET.parse(os.path.join(hls_output_dir, name)).getroot()
            except (ET.ParseError, OSError):
                continue
            for el in root.iter():
                tag = el.tag.split("}")[-1].upper()
                if tag == "CYCLES":
                    v = _num(el.get("value") or el.text)
                    if v is not None:
                        out["cycles"] = int(v)
                        break
            if "cycles" in out:
                break

    # 1) XML 报告里的其它数值标签（有则用之）
    if os.path.isdir(hls_output_dir):
        for name in sorted(os.listdir(hls_output_dir)):
            if not name.lower().endswith(".xml"):
                continue
            p = os.path.join(hls_output_dir, name)
            try:
                root = ET.parse(p).getroot()
            except (ET.ParseError, OSError):
                continue
            for el in root.iter():
                tag = el.tag.split("}")[-1]
                txt = (el.text or "").strip()
                if not txt:
                    continue
                key = tag.lower().replace("-", "_")
                val = _num(txt)
                if val is None:
                    continue
                if "area" in key and "area" not in out:
                    out["area"] = val
                elif "period" in key or "delay" in key:
                    if out.get("target_clock_ns") is None and "target" in key:
                        out["target_clock_ns"] = val
                    elif out.get("achieved_clock_ns") is None:
                        out["achieved_clock_ns"] = val
                elif "register" in key and out.get("registers") is None:
                    out["registers"] = int(val)
                elif "dsp" in key and out.get("dsps") is None:
                    out["dsps"] = int(val)
                elif ("bram" in key or "ram" in key) and out.get("brams") is None:
                    out["brams"] = int(val)

    # 2) 日志文本兜底。Bambu **对每个函数都打印一遍**这些数字（含 softfloat
    #    的 __float_adde11m52b_1023nih 等内部函数，面积甚至是 inf），
    #    直接取最后一次匹配会拿到内部函数的值。所以先按 top 名字切出
    #    顶层函数那一段再解析。
    if log_text:
        scope = log_text
        if top:
            blocks = [b for b in re.split(r"(?=for function )", log_text)
                      if top in b]
            # 面积在 "Module binding information for function X:" 段，
            # 触发器数在后面的 "Total number of flip-flops in function X" 段，
            # 所以要把所有属于顶层的段落都拼起来。
            if blocks:
                scope = "\n".join(blocks)
        pats = {
            "area": r"[Tt]otal\s+estimated\s+area\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
            "registers": r"[Tt]otal\s+number\s+of\s+flip[-\s]?flops\s*"
                         r"(?:in\s+function\s+\S+\s*)?[:=]\s*([0-9]+)",
            "dsps": r"[Ee]stimated\s+number\s+of\s+DSPs?\s*[:=]\s*([0-9]+)",
        }
        for key, pat in pats.items():
            if key in out:
                continue
            ms = re.findall(pat, scope)
            if ms:
                v = _num(ms[-1])
                # 顶层常报 "Total estimated area: inf"（MUX21 面积未知），
                # 对参考没有意义，退回不含 mux 的那个估计。
                if v is not None and v != float("inf"):
                    out[key] = int(v) if key in ("registers", "dsps") else v

        if out.get("area") is None:
            ms = re.findall(r"Estimated\s+resources\s+area\s*"
                            r"\(no\s+Muxes\s+and\s+address\s+logic\)\s*[:=]\s*"
                            r"([0-9]+(?:\.[0-9]+)?)", scope)
            if ms:
                v = _num(ms[-1])
                if v is not None:
                    out["area"] = v

        # 时序：Bambu 只给 "Minimum slack"，达成周期 = 目标周期 - slack
        if out.get("achieved_clock_ns") is None:
            ms = re.findall(r"[Mm]inimum\s+slack\s*[:=]\s*"
                            r"([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)", scope)
            if ms:
                sl = _num(ms[-1])
                if sl is not None and out.get("target_clock_ns"):
                    out["achieved_clock_ns"] = max(
                        0.0, out["target_clock_ns"] - sl)

    # 3) 老式日志文件名兜底
    if "area" not in out and os.path.isdir(hls_output_dir):
        pats = {
            "area": r"[Tt]otal\s+area\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)",
            "achieved_clock_ns": r"[Ee]stimated\s+clock\s+period\s*[:=]?\s*"
                                 r"([0-9]+(?:\.[0-9]+)?)",
            "registers": r"[Nn]umber\s+of\s+registers?\s*[:=]?\s*([0-9]+)",
            "dsps": r"[Nn]umber\s+of\s+DSPs?\s*[:=]?\s*([0-9]+)",
            "brams": r"[Nn]umber\s+of\s+(?:BRAMs?|RAMs?)\s*[:=]?\s*([0-9]+)",
        }
        logs = [n for n in sorted(os.listdir(hls_output_dir))
                if n.lower().endswith((".log", ".txt"))]
        for name in logs:
            txt = _read(os.path.join(hls_output_dir, name))
            if not txt:
                continue
            for key, pat in pats.items():
                if key in out:
                    continue
                m = re.search(pat, txt)
                if m:
                    v = _num(m.group(1))
                    if v is not None:
                        out[key] = int(v) if key in (
                            "registers", "dsps", "brams") else v
    return out


_MODULE_RE = re.compile(r"^\s*module\s+([A-Za-z_][A-Za-z0-9_$]*)",
                        re.MULTILINE)


def detect_top_module(verilog_path: str, top: str) -> str:
    """从生成的 .v 里找出真正的顶层模块名。

    C 内核编译成 `gpt2_tiny_top`，但 C++ 内核会被 Itanium mangle 成
    `_Z13gpt2_tiny_topiiPf` —— 模块名对不上 `--top-fname`，下游 Yosys
    直接报 "Module `gpt2_tiny_top' not found!"。
    """
    txt = _read(verilog_path)
    names = _MODULE_RE.findall(txt)
    if not names:
        return ""
    if top in names:
        return top
    # 优先选名字里**完整包含** top 的模块（mangled 名会带上 top）
    cands = [n for n in names if top in n]
    if cands:
        # 顶层通常没有实例化别人；带下划线后缀的子模块名更长，取最短的更像顶层
        return sorted(cands, key=len)[0]
    return names[-1] if len(names) == 1 else top


def _find_design_verilog(work: str, top: str) -> List[str]:
    """找出 Bambu 生成的设计 .v。

    注意：Bambu 把顶层 ``<top>.v`` 直接写在**工作目录**里，
    ``HLS_output/`` 下只有器件流程脚本和仿真中间物。所以不能只看 HLS_output。
    """
    cand = os.path.join(work, f"{top}.v")
    if os.path.isfile(cand):
        return [cand]

    hits: List[str] = []
    for d, dirs, files in os.walk(work):
        dirs[:] = [x for x in dirs if x not in _NOT_DESIGN_DIRS]
        for f in files:
            if f.endswith(".v") and not f.endswith("_tb.v") \
                    and "testbench" not in f:
                hits.append(os.path.join(d, f))
    if not hits:
        # 兜底：连仿真目录一起看，至少别报"没有 Verilog"
        for d, _dirs, files in os.walk(work):
            hits += [os.path.join(d, f) for f in files if f.endswith(".v")]
    return sorted(set(hits))


def _mem_refs_for_stub(text: str) -> List[str]:
    """只把**仍被 $readmem 引用**的 .mem 当作需要补占位的目标。

    ROM 改写之后 .mem 名只会出现在 ``if (MEMORY_INIT_file == "x.mem")``
    这种字符串比较里，数据已经烤进常量表，再补占位只会误导。
    """
    names: List[str] = []
    for c in re.findall(r"\$readmem[hb]?\s*\([^;]*?\)", text):
        for n in re.findall(r'"([^"]+)"', c):
            if n not in names:
                names.append(n)
    return names


def _mem_geometry(verilog_text: str, mem_name: str):
    """推断某个 MEMORY_INIT_file 对应存储器的 (n_elements, data_size)。

    Bambu 两种写法都有：模块参数 ``data_size=32`` 和例化覆盖 ``.data_size(32)``。
    解析不出来就返回 None，调用方退回保守默认值。
    """
    idx = verilog_text.find(f'"{mem_name}"')
    if idx < 0:
        return None
    win = verilog_text[max(0, idx - 1200): idx + 1200]
    n = re.search(r"n_elements\s*[=(]\s*(\d+)", win)
    d = re.search(r"data_size\s*[=(]\s*(\d+)", win)
    if not d:
        return None
    return (int(n.group(1)) if n else 1, int(d.group(1)))


def _collect_mem(work: str, verilog: List[str], create_stub: bool):
    """收集 .v 引用的全部 .mem；缺失的按需补全零占位。

    返回 (已存在的 mem 路径, 新建的占位 mem 路径)。
    """
    text = ""
    for v in verilog:
        text += _read(v)
    refs = _mem_refs_for_stub(text)
    if not refs:
        for m in _MEM_REF_RE.finditer(text):
            if m.group(1) not in refs:
                refs.append(m.group(1))

    found: List[str] = []
    stubbed: List[str] = []
    for name in refs:
        p = name if os.path.isabs(name) else os.path.join(work, name)
        if os.path.isfile(p):
            found.append(p)
            continue
        if not create_stub:
            continue
        geo = _mem_geometry(text, name)
        n, width = geo if geo else (1, 32)
        try:
            os.makedirs(os.path.dirname(p) or work, exist_ok=True)
            # $readmemb 一个字符 = 1 bit，占位必须写二进制全 0
            with open(p, "w", encoding="utf-8") as f:
                f.write(("0" * max(width, 1) + "\n") * n)
            stubbed.append(p)
        except OSError:
            continue
    return found, stubbed


def run_yosys_check(verilog_path: str, top_module: str, work_dir: str,
                    timeout: int = 3600, yosys: Optional[str] = None,
                    strict: bool = False) -> tuple:
    """用 Yosys 对 Bambu 生成的 RTL 做**展开+工艺无关优化**校验。

    不做 ``synth_xilinx``/``abc``（那属于 ``rtl_backend.synth`` 的职责，
    而且需要器件库）；这里只回答"这份 .v 是不是一个语法正确、能被真正
    展开成门级的设计"：``read_verilog`` -> ``hierarchy -check -top`` ->
    ``proc`` -> ``opt`` -> ``stat``。

    必须在含 .mem 的目录里跑（``$readmemh`` 用相对路径）。

    返回 ``(ok, log_path, summary)``。
    """
    exe = yosys or shutil.which("yosys")
    if not exe:
        return False, "", "未找到 yosys"

    top = top_module or "top"
    work_dir = os.path.abspath(work_dir)
    # `check -assert` 对第三方生成的 RTL 太严：Bambu 顶层会留一个
    # `OUT_UNBOUNDED_*` 信号没人驱动（有意留空的输出通道），一旦 -assert
    # 就整条流程失败。所以默认只跑 `check` 并把告警计入摘要。
    script = (f"read_verilog {os.path.basename(verilog_path)}\n"
              f"hierarchy -check -top {top}\n"
              "proc\n"
              "opt -fast\n"
              + ("check -assert\n" if strict else "check\n")
              + f"stat -top {top}\n")
    ys_path = os.path.join(work_dir, "yosys_check.ys")
    log_path = os.path.join(work_dir, "yosys_check.log")
    try:
        with open(ys_path, "w", encoding="utf-8") as f:
            f.write(script)
        with open(log_path, "w", encoding="utf-8") as lg:
            # 脚本用绝对路径：cwd 会切到 work_dir，相对路径会找不到
            proc = subprocess.run([exe, "-s", ys_path], cwd=work_dir,
                                  stdout=lg, stderr=subprocess.STDOUT,
                                  timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, log_path, f"yosys 超时（>{timeout}s）"
    except OSError as e:
        return False, log_path, f"yosys 执行失败: {e}"

    txt = _read(log_path)
    if proc.returncode != 0:
        errs = [ln.strip() for ln in txt.splitlines()
                if ln.strip().upper().startswith("ERROR")]
        return False, log_path, (errs[-1] if errs
                                 else f"yosys 返回码 {proc.returncode}")

    # `stat` 会把每个模块都打一遍（Bambu 产物有 ~1200 个 module）。
    # 顶层往往只是个 wrapper（`_Z..top` 里只有 1 个 submodule），所以单看
    # 顶层没意义；这里把各模块的本地计数累加成整个设计的规模。
    def _stat(block, label):
        m = (re.search(rf"Number of {label}:\s*(\d+)", block)
             or re.search(rf"^\s*(\d+)\s+{label}\s*$", block, re.MULTILINE))
        return int(m.group(1)) if m else 0

    blocks = re.split(r"^=== .*===$", txt, flags=re.MULTILINE)[1:]
    if not blocks:                       # 老版本可能没有 === 分隔
        blocks = [txt]
    totals = {"cells": 0, "wire_bits": 0, "memories": 0, "memory_bits": 0}
    for blk in blocks:
        for key, label in (("cells", "cells"), ("wire_bits", "wire bits"),
                           ("memories", "memories"),
                           ("memory_bits", "memory bits")):
            totals[key] += _stat(blk, label)

    top_blk = ""
    m = re.search(rf"^=== {re.escape(top)}\s*===$", txt, re.MULTILINE)
    if m:
        rest = txt[m.end():]
        nxt = re.search(r"^=== .*===$", rest, re.MULTILINE)
        top_blk = rest[:nxt.start()] if nxt else rest

    summary = [f"modules={len(blocks)}",
               f"cells={totals['cells']}",
               f"wire_bits={totals['wire_bits']}"]
    if totals["memories"]:
        summary.append(f"memories={totals['memories']}")
        summary.append(f"memory_bits={totals['memory_bits']}")
    if top_blk:
        summary.append(f"top_cells={_stat(top_blk, 'cells')}")
    nwarn = len(re.findall(r"used but has no driver|logic loop", txt))
    if nwarn:
        summary.append(f"check_warnings={nwarn}")
    return True, log_path, "yosys 展开通过 " + " ".join(summary)


def run_bambu(kernel_sources, out_dir: str, top_fname: str,
              cfg: Optional[BambuConfig] = None) -> BambuResult:
    """对给定的 HLS C/C++ 源调用 Bambu，收集 Verilog 与报告。

    Parameters
    ----------
    kernel_sources : str | Sequence[str]
        一个或多个 C/C++ 文件路径。多个文件会被合成到同一个设计里
        （Bambu 支持多输入单顶层）。
    out_dir : str
        工作区根目录；实际工作目录为 ``<out_dir>/<cfg.cwd_name>``，
        Bambu 会在其中创建 ``HLS_output/``。
    top_fname : str
        顶层函数名。
    cfg : BambuConfig, optional
    """
    bcfg = cfg or BambuConfig()
    res = BambuResult()
    top = top_fname or bcfg.top_fname
    res.top_module = top

    ver = bambu_version(bcfg.binary)
    if not ver:
        res.errors.append(
            f"未找到可执行的 {bcfg.binary}（Bambu/PandA 未安装或不在 PATH）。"
            f"安装参考 https://github.com/ferrandi/PandA-bambu ；"
            f"注意 PyPI 上的 `bambu` 包与 PandA 无关。")
        return res

    srcs = [kernel_sources] if isinstance(kernel_sources, str) \
        else list(kernel_sources)
    missing = [s for s in srcs if not os.path.isfile(s)]
    if missing:
        res.errors.append(f"源文件不存在: {missing}")
        return res
    if not top:
        res.errors.append("未指定顶层函数名（top_fname）")
        return res

    work = os.path.join(out_dir, bcfg.cwd_name)
    if os.path.isdir(work):
        shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)

    cmd = [bcfg.binary] + [os.path.abspath(s) for s in srcs]
    cmd.append(f"--top-fname={top}")
    if bcfg.device_name:
        cmd.append(f"--device-name={bcfg.device_name}")
    if bcfg.clock_period:
        cmd.append(f"--clock-period={bcfg.clock_period:g}")
    if bcfg.interface:
        cmd.append(f"--generate-interface={bcfg.interface}")
    if bcfg.compiler:
        cmd.append(f"--compiler={bcfg.compiler}")
    if bcfg.soft_float:
        cmd.append("--soft-float")
    if bcfg.experimental_setup:
        cmd.append(f"--experimental-setup={bcfg.experimental_setup}")
    if bcfg.evaluation:
        cmd.append(f"--evaluation={bcfg.evaluation}")
    if bcfg.simulate:
        cmd.append("--simulate")
        if bcfg.simulator:
            cmd.append(f"--simulator={bcfg.simulator}")
    # -O 必须排在 --experimental-setup 之后：setup 会自带 -O0/-Os，
    # 而 gcc 是"最后一个 -O 生效"，所以我们要的级别放最后。
    if bcfg.opt_level:
        cmd.append(bcfg.opt_level)
    # HLS 里 main() 不可综合，明确排除
    cmd.append("-fno-strict-aliasing")
    if bcfg.faithful_rounding:
        cmd.append("-DFAITHFULLY_ROUNDED")
    # 不链接 libm 时 sqrtf/expf/tanhf 等没有对应 functional unit，
    # 会在 function allocation 阶段报
    # "does not exist a functional unit in the resource library"。
    if bcfg.link_libm:
        cmd.append("-lm")
    cmd += list(bcfg.extra_args)

    res.cmd = cmd
    try:
        p = subprocess.run(cmd, cwd=work, capture_output=True, text=True,
                           timeout=bcfg.timeout)
    except subprocess.TimeoutExpired:
        res.errors.append(f"bambu 超时（>{bcfg.timeout}s）")
        return res
    except OSError as e:
        res.errors.append(f"bambu 执行失败: {e}")
        return res

    res.returncode = p.returncode
    out = (p.stdout or "") + (p.stderr or "")
    res.stdout_tail = "\n".join(out.strip().splitlines()[-40:])
    # PandA 2024.10 不落任何指标报告文件，指标只在 stdout 里，
    # 所以我们必须自己把日志存下来。
    log_path = os.path.join(work, "bambu.log")
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(out)
        res.log_path = log_path
    except OSError:
        res.log_path = ""

    hls = os.path.join(work, "HLS_output")
    if not os.path.isdir(hls):
        hls = work
    res.hls_output_dir = hls

    vs = _find_design_verilog(work, top)
    res.verilog = vs
    if vs:
        res.top_verilog = vs[0] if os.path.basename(vs[0]) == f"{top}.v" else vs[0]
        res.top_module = detect_top_module(res.top_verilog, top) or top

    if vs:
        # 先把 initial/$readmemb 烤成常量 case ROM，必须在 _collect_mem
        # 之前，否则占位文件是为已经不存在的 $readmem 生成的。
        if bcfg.synth_cleanup:
            for v in vs:
                rom = make_rom_synthesizable(v, [work])
                res.rom_files_baked.extend(rom.files_baked)
                res.rom_readmem_removed += rom.readmem_removed
                for w in rom.warnings:
                    res.warnings.append(w)
            if res.rom_readmem_removed:
                res.warnings.append(
                    f"ROM 合成化改写：{res.rom_readmem_removed} 处 "
                    f"initial/$readmem -> 常量 case，固化 "
                    f"{len(set(res.rom_files_baked))} 个 .mem")
            left: Dict[str, int] = {}
            for v in vs:
                try:
                    with open(v, encoding="utf-8", errors="replace") as f:
                        left = scan_simulation_only(f.read())
                except OSError:
                    continue
            if left:
                res.unsynthesizable = left
                res.warnings.append(
                    "改写后仍有仿真专用构造: "
                    + ", ".join(f"{k}x{v}" for k, v in sorted(left.items())))

        found_mem, stubbed_mem = _collect_mem(work, vs, bcfg.mem_stub)
        res.mem_files = found_mem + stubbed_mem
        res.mem_stubbed = stubbed_mem
        if stubbed_mem:
            res.warnings.append(
                f"为 {len(stubbed_mem)} 个缺失的初始化文件生成了全零占位: "
                + ", ".join(os.path.basename(p) for p in stubbed_mem))

        rep = parse_bambu_report(hls, log_text=out, top=top)
        res.area = rep.get("area")
        res.achieved_clock_ns = rep.get("achieved_clock_ns")
        res.target_clock_ns = rep.get("target_clock_ns", bcfg.clock_period)
        res.registers = rep.get("registers")
        res.dsps = rep.get("dsps")
        res.brams = rep.get("brams")
        res.cycles = rep.get("cycles")
        for cand in os.listdir(hls) if os.path.isdir(hls) else []:
            if cand.lower().endswith(".xml"):
                res.report_path = os.path.join(hls, cand)
                break

    if p.returncode != 0:
        res.errors.append(
            f"bambu 返回码 {p.returncode}；见日志 {res.log_path or work}")
        # 分配阶段失败的典型报错（缺 -lm / 没有 FU），摘出来便于定位
        for line in out.splitlines():
            if "functional unit" in line or "not completely allocated" in line:
                res.errors.append(line.strip()[:300])
                break
    elif not res.verilog:
        res.errors.append(f"bambu 未生成 Verilog；输出目录 {work}")
    else:
        res.ok = True

    if res.ok:
        if all(x is None for x in (res.area, res.achieved_clock_ns,
                                   res.registers, res.dsps, res.cycles)):
            res.warnings.append(
                "未能在报告中解析出面积/时序指标（不影响 Verilog 产物）")
    return res
