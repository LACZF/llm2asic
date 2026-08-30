"""rtl_backend subpackage.

`run()` 编排完整 RTL 后端流程：
Parser -> Quantizer -> 黄金参考 -> RTL 生成 -> 仿真 -> 逐位比对报告。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..config import CompileConfig
from .reference import IntModel
from .verilog import generate

__all__ = ["RTLResult", "run"]


@dataclass
class RTLResult:
    """RTL 后端构建与验证的结果。"""
    out_dir: str = ""
    logits_match: int = 0
    logits_total: int = 0
    worst_abs_err: int = 0
    bit_exact: bool = False
    sim_ran: bool = False
    gold_path: str = ""
    sim_logits_path: str = ""
    report_path: str = ""
    errors: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "out_dir": self.out_dir,
            "logits_match": self.logits_match,
            "logits_total": self.logits_total,
            "worst_abs_err": self.worst_abs_err,
            "bit_exact": self.bit_exact,
            "sim_ran": self.sim_ran,
            "gold_path": self.gold_path,
            "sim_logits_path": self.sim_logits_path,
            "errors": self.errors,
        }


def _default_tokens(seq: int, vocab: int) -> np.ndarray:
    """确定性 token 集：覆盖多个不同 token 且不越界。"""
    return np.array([(3 * i + 7) % vocab for i in range(seq)], dtype=np.int64)


def _build_gold(qm, tokens: np.ndarray, out_dir: str) -> np.ndarray:
    """用 IntModel 生成黄金 logits。

    关键：RoPE 位置取 token 的序列下标 ``pos = i``（与 RTL 顶层 ``tokk`` 一致），
    而非 token 标识符本身。KV 缓存跨 token 累积（自回归 decode）。
    """
    qw = dict(qm.engines)
    qw["wte_q"] = qm.wte_q
    for k, v in qm.gammas.items():
        qw[k] = v
    m = IntModel(qw, qm.luts, qm.config)
    seq = int(qm.config["max_seq_len"])
    gold = np.array([m.run_decode_step(int(tokens[i]), i) for i in range(seq)],
                    dtype=np.int64)
    path = os.path.join(out_dir, "gold.npy")
    np.save(path, gold)
    return gold


def _run_tool(cmd: list, cwd: str, what: str) -> None:
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError(
            f"找不到工具 '{cmd[0]}'。RTL 仿真需要安装 Icarus Verilog，"
            f"并确保 iverilog/vvp 在 PATH 中。")
    if proc.returncode != 0:
        raise RuntimeError(
            f"{what} 失败 (返回码 {proc.returncode}):\n{proc.stdout}\n{proc.stderr}")


def run(model_path: str,
        config: Optional[CompileConfig] = None,
        tokens: Optional[np.ndarray] = None) -> RTLResult:
    """执行完整 RTL 后端流程并返回验证报告。

    Parameters
    ----------
    model_path: str
        模型描述文件路径（YAML）。
    config: CompileConfig, optional
        编译配置；默认使用全默认值。
    tokens: np.ndarray, optional
        输入 token 序列（长度 <= max_seq_len）。默认使用确定性序列。
    """
    cfg = config or CompileConfig(model_path=model_path)
    out = cfg.out_dir
    os.makedirs(out, exist_ok=True)
    result = RTLResult(out_dir=out)

    # 惰性导入，避免 parser/quantizer <-> rtl_backend 循环依赖
    from ..parser.builder import run_from_path
    from ..quantizer.pipeline import run as quantizer_run

    try:
        # 1. Parser -> GraphIR
        llm_ir = run_from_path(model_path)

        # 2. Quantizer -> QuantizedModel + ROM
        qm = quantizer_run(llm_ir, cfg, out)

        seq = int(qm.config["max_seq_len"])
        vocab = int(qm.config["vocab_size"])
        toks = tokens
        if toks is None:
            toks = _default_tokens(seq, vocab)
        toks = np.asarray(toks, dtype=np.int64).reshape(-1)
        if toks.shape[0] > seq:
            raise ValueError(
                f"token 序列长度 {toks.shape[0]} 超过 max_seq_len={seq}")

        # 3. 黄金参考（低位定点逐位一致）
        gold = _build_gold(qm, toks, out)
        result.gold_path = os.path.join(out, "gold.npy")

        # 4. 生成 RTL 并复制权重 ROM
        modname = generate(qm, qm.config, out, toks)
        rdir = os.path.join(out, "rtl")
        src = os.path.join(out, "quantizer", "weights_rom")
        for fn in sorted(os.listdir(src)):
            if fn.endswith(".mem"):
                shutil.copy(os.path.join(src, fn), os.path.join(rdir, fn))

        # 5. 仿真（iverilog + vvp）
        if cfg.enable_sim:
            gemv_files = sorted(fn for fn in os.listdir(rdir)
                                if fn.startswith("gemv_") and fn.endswith(".sv"))
            _run_tool(["iverilog", "-g2012", "-o", "sim.vvp",
                       f"{modname}.sv", *gemv_files,
                       "rmsnorm.sv", "attn.sv", "sim_tb.sv"],
                      rdir, what="iverilog 编译")
            proc = subprocess.run(["vvp", "sim.vvp"], cwd=rdir,
                                  capture_output=True, text=True)
            if proc.returncode != 0 or "SIM_DONE" not in proc.stdout:
                raise RuntimeError(
                    f"仿真未正常完成:\n{proc.stdout}\n{proc.stderr}")

            sim_path = os.path.join(rdir, "sim_logits.txt")
            sim = np.array([int(float(x))
                            for x in open(sim_path).read().split()],
                           dtype=np.int64).reshape(seq, -1)
            n_tokens, n_vocab = sim.shape
            result.sim_logits_path = sim_path
            result.sim_ran = True
            if n_vocab != vocab:
                raise ValueError(
                    f"仿真 logits 列数 {n_vocab} 与 vocab={vocab} 不符")

            flat_sim = sim.reshape(-1)
            flat_gold = gold.reshape(-1)[: flat_sim.size]
            result.logits_total = int(flat_sim.size)
            result.logits_match = int(
                np.count_nonzero(flat_sim == flat_gold))
            result.worst_abs_err = int(
                np.max(np.abs(flat_sim - flat_gold))) if flat_sim.size else 0
            result.bit_exact = bool(
                result.logits_match == result.logits_total)
    except Exception as e:  # noqa: BLE001  —— 上报给 CLI，不在此吞掉详情
        result.errors.append(f"{type(e).__name__}: {e}")

    # 6. 报告
    report_path = os.path.join(out, "test_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)
    result.report_path = report_path

    if not result.errors:
        msg = (f"bit-exact (worst={result.worst_abs_err})"
               if result.bit_exact else
               f"{result.logits_match}/{result.logits_total} match, "
               f"worst={result.worst_abs_err}")
        print(f"[rtl_backend] logits {msg}")
    return result
