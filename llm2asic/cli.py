# llm2asic/cli.py
"""命令行入口：`llm2asic build ...` 端到端编译 LLM -> RTL + 仿真验证。"""

from __future__ import annotations

import argparse
import os
import sys

from .config import HlsConfig, load_config
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


def _hls(args) -> int:
    """GraphIR -> HLS C++(/ONNX) -> Bambu -> Verilog。

    与 `build` 的原生 RTL 路径并列，独立开关。
    """
    from .hls_backend import HlsConfig as HlsBackendConfig
    from .hls_backend import build_hls
    from .hls_backend.bambu import BambuConfig
    from .hls_backend.c_kernel import CKernelConfig
    from .parser.builder import run_from_path

    cfg = load_config(args.config, model_path=args.model,
                      out_dir=args.out if args.out else None)
    if args.out:
        cfg.out_dir = args.out
    h: HlsConfig = cfg.hls

    if args.hls_backend:
        h.backend = args.hls_backend
    if args.pos is not None:
        h.pos = args.pos
    if args.no_bambu:
        h.run_bambu = False
    if args.no_verify:
        h.verify = False
    if args.precision:
        h.precision = args.precision
    if args.device:
        h.bambu.device_name = args.device
    if args.clock_period is not None:
        h.bambu.clock_period = args.clock_period
    if args.compiler:
        h.bambu.compiler = args.compiler
    if args.evaluate:
        h.bambu.evaluation = args.evaluate
    if args.simulate:
        h.bambu.simulate = True
    if args.yosys_check is not None:
        h.yosys_check = args.yosys_check
    if args.yosys:
        h.yosys = args.yosys
    if args.pipeline_ii is not None:
        h.pipeline_ii = args.pipeline_ii

    try:
        ir = run_from_path(cfg.model_path)
    except Exception as e:  # noqa: BLE001
        print(f"[llm2asic] error: 解析模型失败: {type(e).__name__}: {e}",
              file=sys.stderr)
        return 1

    out_dir = os.path.join(cfg.out_dir, "hls")
    bcfg = BambuConfig(
        binary=h.bambu.binary,
        device_name=h.bambu.device_name,
        clock_period=h.bambu.clock_period,
        compiler=h.bambu.compiler,
        opt_level=h.bambu.opt_level,
        interface=h.bambu.interface,
        soft_float=h.bambu.soft_float,
        faithful_rounding=h.bambu.faithful_rounding,
        link_libm=h.bambu.link_libm,
        experimental_setup=h.bambu.experimental_setup,
        evaluation=h.bambu.evaluation,
        simulate=h.bambu.simulate,
        simulator=h.bambu.simulator,
        mem_stub=h.bambu.mem_stub,
        synth_cleanup=h.bambu.synth_cleanup,
        timeout=h.bambu.timeout,
    )
    kcfg = CKernelConfig(precision=h.precision, n_buffers=h.n_buffers,
                         pipeline=h.pipeline_ii)
    hcfg = HlsBackendConfig(
        backend=h.backend, out_dir=out_dir, pos=h.pos,
        precision=h.precision,
        hls4ml_precision=h.hls4ml_precision,
        hls4ml_reuse_factor=h.hls4ml_reuse_factor,
        hls4ml_io_type=h.hls4ml_io_type,
        run_bambu=h.run_bambu, verify=h.verify, rel_tol=h.rel_tol,
        yosys_check=h.yosys_check, yosys=h.yosys,
        yosys_timeout=h.yosys_timeout,
        bambu=bcfg, ckernel=kcfg,
    )

    res = build_hls(ir, hcfg)
    for w in res.warnings:
        print(f"[llm2asic] warn: {w}", file=sys.stderr)
    for e in res.errors:
        print(f"[llm2asic] error: {e}", file=sys.stderr)
    if not res.ok:
        return 1

    print(f"[llm2asic] PASS: {res.summary()}")
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

    p_hls = sub.add_parser(
        "hls", help="GraphIR -> HLS C++(/ONNX) -> Bambu -> Verilog")
    p_hls.add_argument("--model", required=True, help="模型描述 YAML 路径")
    p_hls.add_argument("--config", help="编译配置 YAML 路径（可选）")
    p_hls.add_argument("--out", help="输出目录")
    p_hls.add_argument("--hls-backend", dest="hls_backend",
                       choices=["native", "hls4ml", "onnx"],
                       help="HLS 路径（覆盖配置）")
    p_hls.add_argument("--pos", type=int,
                       help="编译期已知的位置索引（decode 第 pos 步）")
    p_hls.add_argument("--no-bambu", action="store_true",
                       help="只产出 HLS C++，不调用 Bambu")
    p_hls.add_argument("--no-verify", action="store_true",
                       help="跳过 C 内核与 numpy 参考的数值比对")
    p_hls.add_argument("--precision", choices=["float", "double"],
                       help="native C 内核精度")
    p_hls.add_argument("--device", help="Bambu 器件名，如 xc7a100t-1csg324-VVD")
    p_hls.add_argument("--clock-period", type=float,
                       help="Bambu 目标时钟周期（ns）")
    p_hls.add_argument("--compiler", help="Bambu 编译器，如 I386_CLANG16")
    p_hls.add_argument("--evaluate",
                       help="Bambu 评估项，如 PERIOD,AREA,REGISTERS,DSPS,BRAMS")
    p_hls.add_argument("--simulate", action="store_true",
                       help="让 Bambu 跑一次仿真验证")
    p_hls.add_argument("--no-yosys-check", dest="yosys_check",
                       action="store_false", default=None,
                       help="跳过 Bambu 产物的 Yosys 展开检查")
    p_hls.add_argument("--yosys", help="yosys 可执行文件路径")
    p_hls.add_argument("--pipeline-ii", type=int,
                       help="`#pragma HLS PIPELINE II=N`；0(默认)=不发该 pragma")
    p_hls.set_defaults(func=_hls)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
