"""Yosys 综合脚本生成与执行（支持 FPGA 与任意 ASIC 标准单元库）。

- backend=fpga：``synth_xilinx``（Xilinx 器件，默认 xc7）。
- backend=asic：``abc -liberty`` + ``dfflibmap`` 标准单元流程，
  liberty 路径通过 ``config.synth.liberty`` 指定（PDK 无关）。

综合在 RTL 目录内运行（``$readmemh`` 使用相对路径），
脚本与网表输出到 ``<out_dir>/synth/``。
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Optional

from ..config import CompileConfig


def discover_liberty(cfg: CompileConfig) -> str:
    """返回可用的 ASIC .lib 路径：显式配置优先，其次常用探测路径。"""
    if cfg.synth.liberty:
        return cfg.synth.liberty
    env = os.environ.get("PDK_LIBERTY")
    if env:
        return env
    pdk = cfg.synth.pdk or "sky130hd"
    # 常见安装布局；优先主标准单元库（sky130_fd_sc_hd__* / *_tt_*）
    import glob
    libdir = f"/home/user/git/eda/back-end/OpenROAD-flow-scripts/flow/platforms/{pdk}/lib"
    preferred = sorted(glob.glob(os.path.join(libdir, "sky130_fd_sc_*_tt_*.lib")))
    if preferred:
        return preferred[0]
    hits = sorted(glob.glob(os.path.join(libdir, "*.lib")))
    if hits:
        return hits[0]
    return ""


def gen_synth_script(cfg: CompileConfig, rtl_dir: str) -> str:
    """生成 Yosys 综合脚本（文本），保存于 synth/ 目录，供 yosys -s 使用。

    Parameters
    ----------
    cfg : CompileConfig
        编译配置；``cfg.synth.backend`` 决定流程（fpga/asic）。
    rtl_dir : str
        含 .sv / .mem 的 RTL 目录（脚本在此目录内运行）。
    """
    sv_files = sorted(
        f for f in os.listdir(rtl_dir)
        if f.endswith(".sv") and f != "sim_tb.sv")
    top = next(f for f in sv_files if f.endswith("_accel.sv"))
    top_mod = os.path.splitext(top)[0]
    gemv = [f for f in sv_files if f.startswith("gemv_") and f.endswith(".sv")]
    others = [f for f in sv_files
              if f not in ([top] + gemv) and f.endswith(".sv")]
    read = (f"read_verilog -sv {' '.join(sv_files)}\n"
            if not gemv else
            f"read_verilog -sv {top}\n"
            + f"read_verilog -sv {' '.join(gemv)}\n"
            + (f"read_verilog -sv {' '.join(others)}\n" if others else ""))

    s = cfg.synth
    if s.backend == "asic":
        lib = discover_liberty(cfg)
        if not lib:
            raise ValueError(
                "SynthConfig.liberty 未设置（backend=asic 需要标准单元 .lib 路径）")
        if not os.path.exists(lib):
            raise ValueError(f"liberty 文件不存在: {lib}")
        script = f'''# LLM2ASIC Yosys 综合（ASIC 标准单元，PDK: {s.pdk or '自定义'}）
# liberty: {lib}

read_liberty -lib {lib}
{read}hierarchy -top {top_mod}
proc
opt
memory -nomap
opt -fast
techmap
abc -liberty {lib}
dfflibmap -liberty {lib}
clean
stat -liberty {lib}
write_verilog -noattr ../synth/netlist.v
'''
    elif s.backend == "fpga":
        family = s.family or "xc7"
        script = f'''# LLM2ASIC Yosys 综合（FPGA，family={family}）
# 在 RTL 目录运行：cd {rtl_dir} && yosys -s ../synth/synth.ys

{read}hierarchy -top {top_mod}
synth_xilinx -top {top_mod} -family {family} -flatten -nowidelut

write_verilog -noattr ../synth/netlist.v
tee -o ../synth/util_report.txt stat -tech xilinx
'''
    else:
        raise ValueError(f"未知 synth.backend: {s.backend!r}（可选 fpga/asic）")
    return script


def run_synth(cfg: CompileConfig, out_dir: str,
              yosys: Optional[str] = None) -> Optional[str]:
    """在 RTL 目录内执行综合，返回网表路径（失败抛 RuntimeError）。

    Parameters
    ----------
    cfg : CompileConfig
        编译配置（含 synth 节与 out_dir）。
    out_dir : str
        构建输出目录（rtl/ 与 synth/ 所在）。
    yosys : str, optional
        yosys 可执行文件；默认取 PATH 中的 yosys。
    """
    yosys = yosys or shutil.which("yosys")
    if not yosys:
        raise RuntimeError("未找到 yosys，请先安装 OpenROAD Yosys")
    out_dir = os.path.abspath(out_dir)
    rtl_dir = os.path.join(out_dir, "rtl")
    synth_dir = os.path.join(out_dir, "synth")
    os.makedirs(synth_dir, exist_ok=True)

    script = gen_synth_script(cfg, rtl_dir)
    script_path = os.path.join(synth_dir, "synth.ys")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script)

    log_path = os.path.join(synth_dir, "synth.log")
    with open(log_path, "w", encoding="utf-8") as lg:
        proc = subprocess.run([yosys, "-s", script_path], cwd=rtl_dir,
                              stdout=lg, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        tail = "".join(open(log_path, "r", errors="replace").readlines()[-40:])
        raise RuntimeError(
            f"Yosys 综合失败（返回码 {proc.returncode}）。日志: {log_path}\n{tail}")

    netlist = os.path.join(synth_dir, "netlist.v")
    if not os.path.exists(netlist) or os.path.getsize(netlist) == 0:
        raise RuntimeError(f"综合未产出网表: {netlist}")
    return netlist