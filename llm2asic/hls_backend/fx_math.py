# llm2asic/hls_backend/fx_math.py
"""定点数学层：可综合、无浮点、无除法器的超越函数与倒数。

`verilog_emit.py` 把本模块生成的常量表和运算结构**原样**翻译成 Verilog，
因此这里的每个函数都对应 RTL 里的一段可综合逻辑：

- 数据格式：signed Q(DW-FR).FR，DW 默认 32、FR 默认 24（范围 ±128）；
- 超越函数 exp/sin/cos/tanh 与 1/sqrt、1/x 全部用**分段线性查表 + 范围归约**
  实现。表的自变量区间取 2 的幂、段数取 2 的幂，于是段号由**纯移位**得到，
  不需要除法器，综合友好；
- 插值分数取 `frac_bits` 位，乘法在 2×DW 位宽内完成后再右移回 FR，
  故全流程无精度损失地保持 Q 格式。

为什么不直接用 `real` / `$exp` / `$sqrt`：iverilog 对 `1.0 << FR` 之类实数
移位直接报错，yosys 连 `real` 函数端口都解析不过（`syntax error,
unexpected TOK_REAL`）。$exp/$sin/$cos/$tanh/$sqrt/$ln 属于 yosys 无法映射
到门级实现的数学单元。所以本层是"要让 yosys 真正综合出网表"的必要条件，
不是精度妥协的产物。

范围归约（消除表值溢出与插值进位溢出）：

- exp(x)  = exp(r) * 2^k,  k = round(x/ln2),  r = x - k*ln2 ∈ [-ln2/2, ln2/2]
- sin(x)  = sin(r),        k = round(x/2π),   r = x - k*2π  ∈ [-π, π]
- cos(x) 同上
- 1/sqrt(v) = T(v/2^e) * 2^(-e/2),  e 取**偶数**（否则 2^(-e/2) 是无理数，
  硬件无法移位实现），v/2^e ∈ [1,4)，故自变量 v/2^(e+2) ∈ [0.25,1)
- 1/v 同上，表为 0.25/t

`frac_bits` 与表长的关系：段宽约为 2^shift，`frac_bits` 取 20 留足乘积余量。
误差随表长每加 1 位收敛约 4 倍（分段线性插值的二次收敛），实测：
exp 7.6e-6→1.9e-6→6.0e-7，sin/cos 1.2e-4→3.1e-5→7.6e-6。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

__all__ = [
    "FxMathError", "FxFormat", "LutTable", "build_lut",
    "ln2", "two_pi",
    "fx_exp", "fx_sin", "fx_cos", "fx_tanh", "fx_invsqrt", "fx_recip",
    "fx_silu", "fx_gelu", "fx_rope_pair",
    "default_luts",
    "ln2_q", "ln2_inv_q", "two_pi_q", "two_pi_inv_q", "rope_inv_freq",
]

# 范围归约用到的常量。RTL 里必须是**量化后**的整数常量，否则圈数 k 可能与
# Python 差 1，残差就差一个周期、结果差 2 倍。
ln2 = float(np.log(2.0))
two_pi = float(2.0 * np.pi)


def ln2_q(fmt: "FxFormat") -> int:
    return int(fmt.quantize(ln2)[()])


def ln2_inv_q(fmt: "FxFormat") -> int:
    return int(fmt.quantize(1.0 / ln2)[()])


def two_pi_q(fmt: "FxFormat") -> int:
    return int(fmt.quantize(two_pi)[()])


def two_pi_inv_q(fmt: "FxFormat") -> int:
    return int(fmt.quantize(1.0 / two_pi)[()])


class FxMathError(RuntimeError):
    """定点数学层参数非法。"""


@dataclass(frozen=True)
class FxFormat:
    """定点数据格式：signed Q(DW-FR).FR。"""

    data_width: int = 32
    frac_bits: int = 24

    def __post_init__(self) -> None:
        if self.data_width < 16:
            raise FxMathError(f"data_width 过小: {self.data_width}")
        if not 8 <= self.frac_bits < self.data_width - 2:
            raise FxMathError(
                f"frac_bits={self.frac_bits} 与 data_width={self.data_width} 不兼容")
        if self.data_width - self.frac_bits < 1:
            raise FxMathError("缺少整数位，无法表示数值范围")

    @property
    def scale(self) -> int:
        return 1 << self.frac_bits

    @property
    def limit(self) -> int:
        """饱和上界（对称）。"""
        return 1 << (self.data_width - 1)

    def quantize(self, x) -> np.ndarray:
        """实数 -> 定点整数。"""
        return np.round(np.asarray(x, dtype=np.float64) * self.scale).astype(np.int64)

    def dequantize(self, q) -> np.ndarray:
        """定点整数 -> 实数。"""
        return np.asarray(q, dtype=np.float64) / self.scale

    def saturate(self, q):
        """饱和到 [-limit, limit)。"""
        return np.clip(q, -self.limit, self.limit - 1)

    def rshift(self, q, bits):
        """带四舍五入的右移（负数也正确；直接 >> 会向下取整把小数抹成 0）。

        `bits` 可以是标量或与 `q` 同形的数组（逐元素移位）。
        """
        q = np.asarray(q, dtype=np.int64)
        b = np.asarray(bits, dtype=np.int64)
        b = np.broadcast_to(b, q.shape)
        if b.size and np.any(b < 0):
            raise FxMathError(f"rshift 只支持右移，收到负移位量 {bits}")
        # 半 LSB 进位；移位量 0 时不进位（否则会凭空 +0.5 LSB）
        half = np.where(b > 0, (1 << (b - 1)) if b.size else 0, 0)
        pos = np.where(q >= 0, (q + half) >> b, -((-q + half) >> b))
        return np.where(b == 0, q, pos)

    def mul(self, a, b):
        """Q.FR * Q.FR -> Q.FR（含四舍五入与饱和）。"""
        return self.saturate(self.rshift(np.asarray(a, np.int64) * np.asarray(b, np.int64),
                                         self.frac_bits))


@dataclass(frozen=True)
class LutTable:
    """分段线性表：区间 [lo, hi) 均分为 2**nbits 段。

    段号 `idx = (x - lo_q) >> shift`，段内分数取 `frac_bits` 位。
    区间端点与段数都取 2 的幂，保证索引只需移位。
    """

    name: str
    lo: float
    hi: float
    nbits: int
    values: np.ndarray          # 共 2**nbits + 1 个 Q.FR 采样值
    lo_q: int
    shift: int
    frac_bits: int = 20

    @property
    def nseg(self) -> int:
        return 1 << self.nbits

    @property
    def depth(self) -> int:
        return self.nseg + 1

    def lookup(self, x_q, fmt: FxFormat):
        """定点输入 -> 定点输出（Q.FR）。"""
        u = np.asarray(x_q, dtype=np.int64) - self.lo_q
        u = np.clip(u, 0, (1 << (self.shift + self.nbits)) - 1)
        idx = np.clip(u >> self.shift, 0, self.nseg - 1)
        off = u - (idx << self.shift)
        if self.shift < self.frac_bits:
            frac = off << (self.frac_bits - self.shift)
        else:
            frac = off >> (self.shift - self.frac_bits)
        d = self.values[idx + 1] - self.values[idx]
        return self.values[idx] + ((d * frac) >> self.frac_bits)


def build_lut(name: str, fn: Callable[[np.ndarray], np.ndarray],
              lo: float, hi: float, nbits: int, fmt: FxFormat,
              extend_first: bool = False, min_index: int = 0) -> LutTable:
    """构造一张分段线性表。

    `lo`/`hi` 需满足 (hi-lo) 是 2 的幂。若 `extend_first` 为真，则函数在 `lo`
    处无定义时（如 1/sqrt(0)、1/0），首项按 `lo + (hi-lo)/nseg` 处的取值延拓。

    `min_index` 声明自变量实际使用区间所对应的最小下标：调用方保证自变量
    落在 `[lo + min_index*(hi-lo)/nseg, hi)`，因此下标 < `min_index` 的项
    永不寻址。这些项若超出定点范围（例如 1/t 在 t 很小时发散）会被钳到可
    表示上界，不影响功能。
    """
    if not (hi > lo):
        raise FxMathError(f"{name}: 非法区间 [{lo}, {hi})")
    width = hi - lo
    if abs(np.log2(width) - round(np.log2(width))) > 1e-9:
        raise FxMathError(f"{name}: 区间宽度 {width} 必须是 2 的幂")
    nseg = 1 << nbits
    edges = np.linspace(lo, hi, nseg + 1)
    # 在 lo 处求值可能溢出（1/sqrt(0)、1/0），这是预期内的，由 extend_first 处理
    with np.errstate(divide="ignore", invalid="ignore"):
        ys = np.asarray(fn(edges), dtype=np.float64)
        if not np.all(np.isfinite(ys)):
            if not extend_first:
                raise FxMathError(f"{name}: 表内出现非有限值，且未启用 extend_first")
            ys = ys.copy()
            ys[0] = fn(np.asarray([lo + width / nseg]))[0]
    if min_index < 0 or min_index > nseg:
        raise FxMathError(f"{name}: min_index={min_index} 越界")
    values = np.round(ys * fmt.scale).astype(np.int64)
    # 越界判定必须在**量化域**做：实数 128 在 Q8.24 下就已经溢出。
    if np.max(np.abs(values[min_index:])) >= fmt.limit:
        raise FxMathError(f"{name}: 量化后表值超出定点范围")
    # 不可寻址的低位项（如 1/t 在 t 很小时发散）钳到可表示上界
    if min_index > 0 and np.max(np.abs(values[:min_index])) >= fmt.limit:
        values = values.copy()
        values[:min_index] = np.clip(values[:min_index], -fmt.limit, fmt.limit - 1)
    shift = fmt.frac_bits + int(round(np.log2(width))) - nbits
    if shift < 0:
        raise FxMathError(f"{name}: 表长 {nseg} 相对区间宽度 {width} 过短")
    return LutTable(name=name, lo=lo, hi=hi, nbits=nbits, values=values,
                    lo_q=int(round(lo * fmt.scale)), shift=shift)


# --------------------------------------------------------------------------
# 表构建（与 Verilog 发射器共用同一份常量）
# --------------------------------------------------------------------------

def default_luts(fmt: FxFormat, lut_bits: int = 9, tanh_bits: int = 9,
                 recip_bits: int = 9) -> dict:
    """构造全部数学常量表。

    区间端点全部取 2 的幂：
    - exp 的残差落在 [-ln2/2, ln2/2] ⊂ [-1, 1)
    - sin/cos 的残差落在 [-π, π] ⊂ [-4, 4)
    - tanh 输入钳到 [-8, 8)（GELU 的 u = 0.798(x+0.0447x^3) 在 |x|<=4 时 |u|<=5.5）
    - 1/sqrt 与 1/x 的自变量落在 [0.25, 1) ⊂ [0, 1)
    """
    return {
        "EXP": build_lut("EXP", np.exp, -1.0, 1.0, lut_bits, fmt),
        "SIN": build_lut("SIN", np.sin, -4.0, 4.0, lut_bits, fmt),
        "COS": build_lut("COS", np.cos, -4.0, 4.0, lut_bits, fmt),
        "TANH": build_lut("TANH", np.tanh, -8.0, 8.0, tanh_bits, fmt),
        # 1/sqrt(t) 在 t=0 处无定义，首项按 t=1/nseg 延拓（实际不寻址）
        # 自变量恒在 [0.25, 1)，故下标 < nseg/4 的项不可寻址
        "ISQRT": build_lut("ISQRT", lambda t: 0.5 / np.sqrt(t), 0.0, 1.0,
                           recip_bits, fmt, extend_first=True,
                           min_index=1 << (recip_bits - 2)),
        # 0.25/t：自变量 t = m/4，使 1/m = 0.25/t，指数即可为偶数
        "RECIP": build_lut("RECIP", lambda t: 0.25 / t, 0.0, 1.0,
                           recip_bits, fmt, extend_first=True,
                           min_index=1 << (recip_bits - 2)),
    }


# --------------------------------------------------------------------------
# 范围归约 + 表查找
# --------------------------------------------------------------------------

def _scale_pow2(y_q, k, fmt: FxFormat):
    """Q.FR 值乘 2**k，结果仍为 Q.FR（整数移位，保持格式不变）。"""
    k = np.clip(np.asarray(k, dtype=np.int64), -60, 60)
    # 右移用带进位的移位，与 FxFormat.rshift / RTL 的 fx_rshr 一致
    return fmt.saturate(np.where(k >= 0,
                                 y_q << np.maximum(k, 0),
                                 fmt.rshift(y_q, np.maximum(-k, 0))))


def _residual(x, inv_period_q: int, period_q: int, fmt: FxFormat):
    """把定点 x 归约到 [-period/2, period/2)，返回 (残差定点, 圈数 k)。

    k = round(x / period) 用**量化后的倒数常量**算，全程定点，RTL 可逐位复现
    （若这里用浮点 `round(x/period)`，k 一旦差 1，残差就差一个周期，结果差
    2 倍）。两次带进位的右移也刻意保留：RTL 照同样的顺序做，才能逐位一致。
    """
    x = np.asarray(x, dtype=np.int64)
    k_fx = fmt.rshift(x * int(inv_period_q), fmt.frac_bits)   # Q.FR 的 x/period
    k = fmt.rshift(k_fx, fmt.frac_bits)                      # 整数圈数
    k_q = k << fmt.frac_bits                                 # k 的 Q.FR 表示
    return x - fmt.mul(k_q, int(period_q)), k


def _even_exponent(v_q, fmt: FxFormat):
    """取偶数指数 e，使 v = m * 2**e 且 m ∈ [1, 4)。返回 e（整数）。

    偶数指数保证 2**(-e/2) 是整数幂，1/sqrt 与 1/x 都只需整数移位。
    浮点版用 floor(log2(v))；定点版取 |v| 的最高有效位，两者等价。
    """
    a = np.abs(np.asarray(v_q, dtype=np.int64))
    p = np.where(a > 0, (np.floor(np.log2(np.maximum(a, 1)))).astype(np.int64),
                 0) - fmt.frac_bits
    p = np.clip(p, -60, 60)
    return p - (p & 1)


def _mantissa_q(v_q, e, fmt: FxFormat):
    """t = v * 2**(-e) / 4 ∈ [0.25, 1)，返回其 Q.FR 定点表示。

    2**(-e-2) 是 2 的幂，所以这一步就是一次移位，RTL 可逐位复现。
    """
    sh = e + 2
    v_q = np.asarray(v_q, dtype=np.int64)
    return np.where(sh >= 0, fmt.rshift(v_q, np.maximum(sh, 0)),
                    v_q << np.maximum(-sh, 0))


def fx_exp(x, luts: dict, fmt: FxFormat):
    """exp(x)，定点进定点出。结果超出定点范围时饱和。

    注意范围：Q8.24 的上限是 128，故 exp(x) 在 x > ~4.86 时会饱和。本项目
    只在 SiLU 内部调用，且自变量恒取 -|x| <= 0，结果落在 (0, 1]，永不溢出。
    """
    r, k = _residual(x, ln2_inv_q(fmt), ln2_q(fmt), fmt)
    return _scale_pow2(luts["EXP"].lookup(r, fmt), k, fmt)


def fx_sin(x, luts: dict, fmt: FxFormat):
    r, _ = _residual(x, two_pi_inv_q(fmt), two_pi_q(fmt), fmt)
    return luts["SIN"].lookup(r, fmt)


def fx_cos(x, luts: dict, fmt: FxFormat):
    r, _ = _residual(x, two_pi_inv_q(fmt), two_pi_q(fmt), fmt)
    return luts["COS"].lookup(r, fmt)


def fx_tanh(x, luts: dict, fmt: FxFormat):
    """tanh(x)。输入钳到表区间 ±8；|x|>8 时 tanh 已饱和到 1（误差 <2.4e-7）。"""
    x = np.clip(np.asarray(x, dtype=np.int64),
                -(8 << fmt.frac_bits), (8 << fmt.frac_bits) - 1)
    return luts["TANH"].lookup(x, fmt)


def fx_invsqrt(v, luts: dict, fmt: FxFormat):
    """1/sqrt(v)，v > 0。结果超出定点范围时饱和。

    Q8.24 下 1/sqrt(v) 在 v < ~6e-5 时会饱和；norm 的方差远大于该值。
    """
    v = np.maximum(np.atleast_1d(v).astype(np.int64), 1)
    e = _even_exponent(v, fmt)
    t = _mantissa_q(v, e, fmt)                      # ∈ [0.25, 1)
    y = luts["ISQRT"].lookup(t, fmt)                # 已是 Q.FR
    return _scale_pow2(y, -(e // 2), fmt)           # 1/sqrt = (1/sqrt m) * 2**(-e/2)


def fx_recip(v, luts: dict, fmt: FxFormat):
    """1/v，v > 0。结果超出定点范围时饱和。

    Q8.24 下 1/v 在 v < ~0.0078 时会饱和；SiLU 传入的是 1+exp(-|x|) ∈ [1,2]。
    """
    v = np.maximum(np.atleast_1d(v).astype(np.int64), 1)
    e = _even_exponent(v, fmt)
    t = _mantissa_q(v, e, fmt)
    y = luts["RECIP"].lookup(t, fmt)
    # 1/v = (1/m) * 2**(-e)：指数是完整的 -e，不是 -e/2
    return _scale_pow2(y, -e, fmt)


# --------------------------------------------------------------------------
# 激活函数与 RoPE 配对
# --------------------------------------------------------------------------

def fx_silu(x, luts: dict, fmt: FxFormat):
    """SiLU / swish: x * sigmoid(x)。

    x>=0 用 x/(1+exp(-x))；x<0 用 x*exp(x)/(1+exp(x))。这样 exp 的自变量
    恒为 -|x| <= 0，落在 (0,1]，指数结果永不溢出 40 位定点。
    """
    x = np.atleast_1d(x).astype(np.int64)
    t = fx_exp(-np.abs(x), luts, fmt)               # ∈ (0, 1]
    den = t + (1 << fmt.frac_bits)                  # ∈ [1, 2]
    rc = fx_recip(den, luts, fmt)
    num = np.where(x < 0, fmt.mul(x, t), x)
    return fmt.saturate(fmt.mul(num, rc))


def fx_gelu(x, luts: dict, fmt: FxFormat,
            alpha: float = 0.7978845608028654,
            beta: float = 0.044715) -> np.ndarray:
    """tanh 近似的 GELU: 0.5*x*(1+tanh(alpha*(x+beta*x^3)))。"""
    x = np.atleast_1d(x).astype(np.int64)
    a_q = int(fmt.quantize(alpha))
    b_q = int(fmt.quantize(beta))
    one = 1 << fmt.frac_bits
    x3 = fmt.mul(fmt.mul(x, x), x)
    inner = fmt.saturate(x + fmt.mul(x3, b_q))
    u = fmt.mul(a_q, inner)
    th = fx_tanh(u, luts, fmt)
    # 0.5*x*(1+tanh(u))：half = (1+tanh)/2 ∈ [0,1]（已含 0.5），再乘 x 并右移 FR 位
    half = fmt.rshift(one + th, 1)
    prod = x.astype(np.int64) * half.astype(np.int64)      # Q(2*FR)
    return fmt.saturate(fmt.rshift(prod, fmt.frac_bits))


def fx_rope_pair(xa: int, xb: int, ang_q: int, cos_q: int, sin_q: int,
                 fmt: FxFormat):
    """RoPE 单对旋转：返回 (ya, yb)。

    角度 `ang_q` 依赖运行时输入 `pos`，由调用方用编译期常量
    `inv_freq_i = theta**(-2*i/head_dim)` 乘 `pos` 得到，再经 `fx_sin`/
    `fx_cos` 得到 `cos_q`/`sin_q`。此处只做乘加。
    """
    a = fmt.mul(xa, cos_q)
    b = fmt.mul(xb, sin_q)
    c = fmt.mul(xa, sin_q)
    d = fmt.mul(xb, cos_q)
    ya = fmt.saturate(a - b)
    yb = fmt.saturate(c + d)
    return int(ya), int(yb)


def rope_inv_freq(head_dim: int, theta: float, fmt: FxFormat) -> list:
    """RoPE 的每通道倒数频率常量 `theta**(-2*i/head_dim)`，量化成 Q.FR。

    这些只依赖 `head_dim`/`theta`，与 `pos` 无关，可烧进 ROM。
    """
    if head_dim % 2 != 0 or head_dim <= 0:
        raise FxMathError(f"head_dim 必须为正偶数，实际 {head_dim}")
    return [int(fmt.quantize(theta ** (-2.0 * i / head_dim)))
            for i in range(head_dim // 2)]
