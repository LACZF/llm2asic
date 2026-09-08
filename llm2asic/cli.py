# llm2asic/cli.py
"""命令行入口：`llm2asic build ...` 端到端编译 LLM -> RTL + 仿真验证。"""

from __future__ import annotations

import argparse
import os
import sys

from .config import load_config
from .rtl_backend import run

__all__ = ["main"]


def _build(args) -> int:
    cfg = load_config(args.config, model_path=args.model, out_dir=args.out
                      if args.out else None)
    if args.model:
        cfg.model_path = args.model
    if args.out:
        cfg.out_dir = args.out
    if args.no_sim:
        cfg.enable_sim = False
    if args.single_file:
        cfg.single_file = True

    result = run(cfg.model_path, cfg)

    if result.errors:
        for err in result.errors:
            print(f"[llm2asic] error: {err}", file=sys.stderr)
        return 1
    if not result.sim_ran:
        print(f"[llm2asic] RTL 已生成于 {result.out_dir}/rtl（跳过仿真）")
        return 0
    if result.bit_exact:
        print(f"[llm2asic] PASS: logits 逐位一致 "
              f"({result.logits_match}/{result.logits_total})")
        return 0
    print(f"[llm2asic] FAIL: {result.logits_match}/{result.logits_total} "
          f"逐位一致，worst_abs={result.worst_abs_err}",
          file=sys.stderr)
    return 1


def _synth(args) -> int:
    """综合已有 RTL（先保证存在）为门级网表。"""
    from .rtl_backend.synth import run_synth

    cfg = load_config(args.config, model_path=args.model, out_dir=args.out
                      if args.out else None)
    if args.model:
        cfg.model_path = args.model
    if args.out:
        cfg.out_dir = args.out
    if args.backend:
        cfg.synth.backend = args.backend
    if args.liberty:
        cfg.synth.liberty = args.liberty
    if args.pdk:
        cfg.synth.pdk = args.pdk
    if hasattr(args, "single_file") and args.single_file:
        cfg.single_file = True

    # 确保 RTL 已生成
    rtl_dir = os.path.join(cfg.out_dir, "rtl")
    if not (os.path.isdir(rtl_dir) and any(
            f.endswith("_accel.sv") for f in os.listdir(rtl_dir))):
        print(f"[llm2asic] 未找到已生成的 RTL（{rtl_dir}），先生成 RTL…")
        r = run(cfg.model_path, cfg)
        if r.errors:
            for err in r.errors:
                print(f"[llm2asic] error: {err}", file=sys.stderr)
            return 1

    try:
        netlist = run_synth(cfg, cfg.out_dir)
    except Exception as e:  # noqa: BLE001
        print(f"[llm2asic] error: {type(e).__name__}: {e}",
              file=sys.stderr)
        return 1
    print(f"[llm2asic] PASS: 网表 -> {netlist}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="llm2asic",
        description="将大语言模型权重编译为定制化 RTL 硬件电路。")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="端到端编译 + 仿真验证")
    p_build.add_argument("--model", required=True,
                         help="模型描述 YAML 路径")
    p_build.add_argument("--config",
                         help="编译配置 YAML 路径（可选）")
    p_build.add_argument("--out", help="输出目录")
    p_build.add_argument("--no-sim", action="store_true",
                         help="仅生成 RTL，不跑仿真")
    p_build.add_argument("--single-file", action="store_true",
                         help="额外产出合并的单文件 RTL (*_single.sv)")
    p_build.set_defaults(func=_build)

    p_synth = sub.add_parser("synth", help="综合已生成 RTL 为门级网表")
    p_synth.add_argument("--model", required=True,
                         help="模型描述 YAML 路径")
    p_synth.add_argument("--config",
                         help="编译配置 YAML 路径（可选）")
    p_synth.add_argument("--out", help="输出目录")
    p_synth.add_argument("--backend", choices=["fpga", "asic"],
                         help="综合后端（覆盖配置）")
    p_synth.add_argument("--liberty", help="ASIC 标准单元 .lib 路径")
    p_synth.add_argument("--pdk", help="PDK 名（仅用于报告）")
    p_synth.add_argument("--single-file", action="store_true",
                         help="生成时额外产出合并的单文件 RTL (*_single.sv)")
    p_synth.set_defaults(func=_synth)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
