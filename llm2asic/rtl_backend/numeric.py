# llm2asic/rtl_backend/numeric.py
"""整数（定点）推理数值模型 —— RTL 与 numpy 黄金参考共享的"唯一数值真值"。

约定（保证 RTL 与软件黄金逐位一致）：
- 所有激活张量 T 以整数 `T.q` 存储，`T.true = T.q / 2^F`（全局分数位 F）。
- 权重逐行量化：`w_true[o,:] = wq[o,:] * s_row[o]`，`s_row[o] = mant[o] * 2^e[o]`。
- 非线性（rmsnorm 的开方/倒数、silu、softmax 的 exp）由"编译器生成的定点查找表"实现，
  这些 LUT 以 .mem 落盘并由 RTL 读取；numpy 参考用完全相同的 LUT，从而逐位一致。

本模块同时提供：
  gen_luts()     生成确定性定点 LUT（sqrt 倒数 / sigmoid / exp / 除法取整）
  quantize_row() 逐行权重量化
  execute()      对 GraphIR 执行整数推理（生成黄金向量）
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

F = 12            # 全局激活分数位
ACT_BITS = 24     # 激活整型位宽（有符号）


def banker_round_shift(v: int, k: int) -> int:
    """round-half-even 的 v/2^k -> int。与 RTL 逐位一致。"""
    q = v >> k
    rem = v - (q << k)          # 0 .. 2^k-1
    half = 1 << (k - 1)
    if rem > half:
        return q + 1
    if rem == half:
        return q if (q % 2 == 0) else q + 1
    return q


# ---------------------------------------------------------------------------
# LUT 生成（确定性）
# ---------------------------------------------------------------------------

@dataclass
class LUT:
    """一张定点查找表：in 为有符号整数输入（可含偏置/移位），out 为整数。"""
    name: str
    input_bits: int = 16       # 输入整型位宽（有符号），用于裁剪/寻址
    output_bits: int = 24
    table: np.ndarray = None   # 一维 int32 数组
    xmin: int = -32768
    xmax: int = 32767
    # 输入可能需要除以 2^k 再查表（避免表过大）；用 scale_in_shift 表达
    in_shift: int = 0
    # 输出需要乘 2^k（恢复到 2^F 分数量纲）
    out_shift: int = 0

    def to_mem_lines(self, per_line: int = 1) -> list[str]:
        if per_line == 1:
            return [f"{int(v):0{8}x}" for v in self.table]
        lines = []
        for i in range(0, len(self.table), per_line):
            chunk = self.table[i:i + per_line]
            val = 0
            for j, v in enumerate(chunk):
                val |= (int(v) & ((1 << (8 * per_line)) - 1)) << (0)  # 占位
            # 简单：每字一条
            lines.append(f"{int(chunk[0]) & 0xFFFFFFFF:08x}")
        return lines


def _make_rqlut(num_steps: int = 4096, input_max: int = 2**20) -> np.ndarray:
    """生成 y = round(2^24 / x) 的定点倒数 LUT，x 从 1..input_max。

    用折半查找表加速（每个 input 一个入口会过大），这里用线性近似：
    步进 x，取 x 处精确的 2^24/x，允许 RTL 以相邻查表。
    """
    xs = np.arange(1, input_max + 1)
    vals = np.zeros(input_max + 1, dtype=np.int64)
    vals[1:] = np.rint(2.0**24 / xs).astype(np.int64)
    # 为节约 RTL 面积，稀疏化为 2^k 步长，并以线性插值——但为保证逐位一致，
    # 这里做成完整表（compile-time 生成），RTL 面积另由 ArchDesc 权衡。
    return vals


@dataclass
class LUTSet:
    """数值模型所需的全部定点查找表。"""
    rsqrt: np.ndarray = None          # y = round(2^F * 1/sqrt(x)) for x=1..MAX
    sigmoid: np.ndarray = None        # y = round(2^F * sigmoid(x)) for x in inputs
    silu_gate: np.ndarray = None      # y = round(2^F * (x * sigmoid(x))) for x in inputs
    gelu: np.ndarray = None           # y = round(2^F * gelu(x))  GPT-2
    exp_neg: np.ndarray = None        # y = round(2^F * exp(-x)) for x>=0,int
    recip2: np.ndarray = None         # y = round(2^FF / x) 倒数（softmax/除法用）

    gelu_input_bits: int = 12

    @property
    def files(self) -> dict:
        d = {
            "rsqrt.mem": self.rsqrt,
            "sigmoid.mem": self.sigmoid,
            "silu.mem": self.silu_gate,
            "gelu.mem": self.gelu,
            "exp_neg.mem": self.exp_neg,
            "recip.mem": self.recip2,
        }
        return {k: v for k, v in d.items() if v is not None}


def gen_luts(act_bits: int = ACT_BITS, Fbits: int = F, rsqrt_bits: int = 20) -> LUTSet:
    """确定性生成整套定点 LUT。

    Parameters
    ----------
    rsqrt_bits : int
        rsqrt 查找表项数 = 2^rsqrt_bits。默认 20（2^20≈104 万项，精度最高）；
        ASIC 综合为了方便在受限内存下展开，可调小（精度随之下探，
        但 RTL 与黄金参考始终使用同一张表，保持逐位一致）。
    """
    lut = LUTSet()
    SM = 2.0**Fbits

    # 1) sigmoid / silu：输入裁剪到 [-2^(X-1), 2^(X-1)-1]，按 in_shift 分桶
    X = 10                      # sigmoid/silu 输入整数位宽（有符号）
    xs = np.arange(-2**(X-1), 2**(X-1))
    xr = xs.astype(np.float64) / SM          # 真实值
    sig = 1.0 / (1.0 + np.exp(-xr))
    lut.sigmoid = np.rint(sig * SM).astype(np.int64)
    lut.silu_gate = np.rint(xr * sig * SM).astype(np.int64)
    lut.sigmoid_input_bits = X
    lut.silu_input_bits = X

    # 1b) gelu（GPT-2）：tanh 近似，输入裁剪到 [-2^(G-1), 2^(G-1)-1]
    G = 12
    xg = np.arange(-2**(G-1), 2**(G-1))
    xgr = xg.astype(np.float64) / SM
    t = np.tanh(np.sqrt(2.0 / np.pi) * (xgr + 0.044715 * xgr**3))
    lut.gelu = np.rint((0.5 * xgr * (1.0 + t)) * SM).astype(np.int64)
    lut.gelu_input_bits = G

    # 2) rsqrt：x = 归一化后的整数（0..MAX），y = round(2^F * 1/sqrt(x))
    #    项数 = 2^rsqrt_bits + 1（索引 0..2^rsqrt_bits）；默认 2^20+1 与历史一致。
    MAX = 1 << rsqrt_bits
    xv = np.arange(0, MAX + 1, dtype=np.float64)
    rsqrt = np.zeros(MAX + 1, dtype=np.int64)
    rsqrt[1:] = np.rint(SM / np.sqrt(xv[1:])).astype(np.int64)
    lut.rsqrt = rsqrt

    # 3) exp_neg：x = 裁剪非负整数 → y = round(2^F * exp(-x/SM))
    XE = 12
    xe = np.arange(0, 2**XE)
    xer = xe.astype(np.float64) / SM
    lut.exp_neg = np.rint(np.exp(-xer) * SM).astype(np.int64)
    lut.exp_neg_input_bits = XE

    # 4) recip2：y = round(2^(2*F) / x)，x 非负整数（idiv 用）
    RM = 2.0 ** (2 * Fbits)
    xv2 = np.arange(1, 2**15)
    lut.recip2 = np.rint(RM / xv2).astype(np.int64)

    return lut


# ---------------------------------------------------------------------------
# 量化
# ---------------------------------------------------------------------------

@dataclass
class QWeight:
    """量化后的单个权重张量。"""
    name: str
    wq: np.ndarray          # [C_out, C_in] 整型（有符号）
    scale_num: np.ndarray   # [C_out] num[o]（requant 分子）
    requant_shift: int      # out.q[o] = round(acc[o]*num[o] / 2^S)
    bit_width: int
    group_size: int
    row_rename: str = ""
    rom_file: str = ""
    scale_rom_file: str = ""
    bias_q: np.ndarray = None        # [C_out] 定点 bias（可选，GPT-2）
    bias_rom_file: str = ""

    @property
    def c_out(self):
        return self.wq.shape[0]

    @property
    def c_in(self):
        return self.wq.shape[1]


# 全局 requant shift（线性层输出的激活定为 2^F 分数量纲时的固定移位）
REQUANT_S = 16


def _quantize_row(w: np.ndarray, bit_width: int, group_size: int, s_shift: int = REQUANT_S):
    """逐行（等价 group_size=C_in）对称量化。

    返回 wq[int32], num[int]（每个输出行一个 requant 分子），s_shift 固定。
    """
    c_out, c_in = w.shape
    amax = np.abs(w).max(axis=1, keepdims=True).clip(min=1e-9)   # [C_out,1]
    qmax = (1 << (bit_width - 1)) - 1
    s_row = (amax / qmax).reshape(-1)                             # [C_out] 真实 scale
    wq = np.rint(w / s_row[:, None]).clip(-qmax - 1, qmax).astype(np.int64)
    num = np.rint(s_row * (2.0 ** s_shift)).astype(np.int64)
    num = np.clip(num, 1, 2**30)
    return wq, num, s_shift


def quantize_weight(weight: np.ndarray, bit_width: int, group_size: int,
                    name: str = "") -> QWeight:
    """把浮点权重 [C_out, C_in] 量化为逐行定点。"""
    w = np.asarray(weight, dtype=np.float64)
    if w.ndim != 2:
        raise ValueError(f"权重 {name} 应为 2D，得到 {w.shape}")
    wq, num, shift = _quantize_row(w, bit_width, group_size)
    return QWeight(name=name, wq=wq, scale_num=num, requant_shift=shift,
                   bit_width=bit_width, group_size=group_size)
