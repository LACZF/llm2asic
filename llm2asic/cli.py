# llm2asic/cli.py
"""命令行入口：`llm2asic build ...` 端到端编译 LLM -> RTL + 仿真验证。"""

from __future__ import annotations

import argparse
import sys

from .config import load_config
from .rtl_backend import run


def _build(args) -> int:
    cfg = load_config(args.config, model_path=args.model, out_dir=args.out
                      if args.out else None)
    if args.model:
        cfg.model_path = args.model
    if args.out:
        cfg.out_dir = args.out
    if args.no_sim:
        cfg.enable_sim = False

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
    p_build.set_defaults(func=_build)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
