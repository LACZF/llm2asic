"""GraphIR 的定点参考实现。

`fx_decode_step` 按 `float_ref.ref_decode_step` 相同的语义，但所有算子都在
Q.FR 定点域上计算，并只使用 `fx_math` 里那些**可综合**的运算（分段线性表 +
移位 + 一个乘法器）。它有两个用途：

1. 验证定点方案本身的数值精度（对照浮点参考）；
2. 作为 Verilog 发射器的黄金模型：Icarus 仿真拿它生成的激励、
   用 `float_ref` 校验 RTL 输出，或直接逐算子 trace 对拍。

与未来 RTL 的一致性约定
---------------------
* 数据通路一律饱和到 `fmt` 的可表示范围；不饱和的只有点积的**累加器**。
* 除以元素个数 `n` 用编译期常量 `round(2^FR / n)` 实现（乘 + 移位），
  而不是运行时除法器。
* `theta**(-2i/head_dim)` 是编译期常量，运行时只算 `ang = pos * inv_freq`。
"""

from __future__ import annotations

import numpy as np

from ..ir.graph import GraphIR
from ..ir.ops import Op

from .fx_math import (
    FxFormat,
    default_luts,
    fx_cos,
    fx_exp,
    fx_gelu,
    fx_invsqrt,
    fx_recip,
    fx_silu,
    fx_sin,
    rope_inv_freq,
)


class FxRefError(RuntimeError):
    pass


_PASSTHROUGH = {
    Op.PERMUTE.value, Op.TRANSPOSE.value, Op.KV_STORE.value, Op.KV_LOAD.value,
    Op.CONCAT.value, Op.UNFLATTEN.value, Op.GEMM.value, Op.GEMV.value,
    Op.MATMUL.value, Op.BMM.value,
}


def op_type(node) -> str:
    ot = node.op_type
    return ot.value if isinstance(ot, Op) else str(ot)


def _inv_n(n: int, fmt: FxFormat) -> int:
    """编译期常量 round(2^FR / n)：除以元素个数用它，无需运行时除法器。"""
    return int(fmt.quantize(1.0 / n))


def _mean_q(sum_q, n: int, fmt: FxFormat) -> int:
    """`sum_q / n`，`sum_q` 是 Q.FR 的累加结果。用编译期倒数实现。"""
    return int(fmt.rshift(np.int64(sum_q) * _inv_n(n, fmt), fmt.frac_bits))


def _mean_of_sumsq_q(s2, n: int, fmt: FxFormat) -> int:
    """`sum(x_i^2) / n`，`s2` 是 Q(2*FR) 的累加结果，返回 Q.FR。

    先右移 FR 把 Q(2*FR) 降回 Q.FR 再乘倒数：s2 本身已接近 int64 上限，
    直接乘 2^24/n 会溢出。这也正是 RTL 的顺序（宽累加器 -> 移位 -> 乘倒数）。
    """
    lo = int(fmt.rshift(np.int64(s2), fmt.frac_bits))
    return int(fmt.rshift(np.int64(lo) * _inv_n(n, fmt), fmt.frac_bits))


class QuantizedModel:
    """把 GraphIR 的权重一次性量化到 Q.FR，并缓存倒数频率等常量。"""

    def __init__(self, ir: GraphIR, fmt: FxFormat):
        self.ir = ir
        self.fmt = fmt
        self.luts = default_luts(fmt)
        self._w: dict = {}
        for n in ir.nodes:
            for wn in n.weight_names:
                if wn in self._w:
                    continue
                w = ir.weights.get(wn)
                if w is None or w.data is None:
                    raise FxRefError(f"权重 {wn} 缺失数值")
                self._w[wn] = fmt.quantize(np.asarray(w.data, dtype=np.float64))
        cfg = ir.config or {}
        self.head_dim = int(cfg.get("head_dim", 0) or 0)
        theta = float(cfg.get("rope_theta", 10000.0) or 10000.0)
        self.inv_freq = rope_inv_freq(self.head_dim, theta, fmt) \
            if self.head_dim else []

    def W(self, wn: str) -> np.ndarray:
        return self._w[wn]


def fx_decode_trace(qm: QuantizedModel, token: int, pos: int) -> list:
    """返回 `[(节点名, 算子, Q.FR 输出数组)]`，即定点单步 decode 的逐算子结果。"""
    ir, fmt, luts = qm.ir, qm.fmt, qm.luts
    eps = float((ir.config or {}).get("norm_eps", 1e-5) or 1e-5)
    eps_q = int(fmt.quantize(eps))

    by_op: dict = {}
    for n in ir.nodes:
        by_op.setdefault(op_type(n), []).append(n)
    if Op.EMBEDDING.value not in by_op:
        raise FxRefError("GraphIR 缺少 EMBEDDING 节点")
    emb = by_op[Op.EMBEDDING.value]
    prologue = emb[:2] if len(emb) >= 2 else emb[:1]

    vals: dict = {}
    vals[emb[0].outputs[0]] = qm.W(emb[0].weight_names[0])[int(token)].copy()
    if len(prologue) >= 2:
        vals[prologue[1].outputs[0]] = qm.W(prologue[1].weight_names[0])[int(pos)].copy()

    trace: list = []
    for n in ir.nodes:
        if any(n is e for e in prologue):
            continue
        t = op_type(n)
        if t in _PASSTHROUGH:
            continue
        ins = [i for i in n.inputs if i in vals]
        if not ins or not n.outputs:
            continue
        o = n.outputs[0]
        x = vals[ins[0]]
        n_el = int(x.size)

        if t == Op.LINEAR.value:
            acc = (qm.W(n.weight_names[0]).astype(np.int64) @ x.astype(np.int64))
            y = fmt.rshift(acc, fmt.frac_bits)
            for wn in n.weight_names[1:]:
                y = y + qm.W(wn)
        elif t == Op.LAYERNORM.value:
            # 与 float_ref 一致：weight_names[0] 是 gamma（乘），其余是 beta（加）
            mu = _mean_q(int(np.sum(x.astype(np.int64), dtype=np.int64)), n_el, fmt)
            d = x - mu
            s2 = int(np.sum(d.astype(np.int64) ** 2, dtype=np.int64))     # Q(2FR)
            var = _mean_of_sumsq_q(s2, n_el, fmt)
            y = fmt.mul(d, fx_invsqrt(np.asarray([var + eps_q]), luts, fmt)[0])
            y = fmt.mul(y, qm.W(n.weight_names[0]))
            for wn in n.weight_names[1:]:
                y = y + qm.W(wn)
        elif t == Op.RMSNORM.value:
            s2 = int(np.sum(x.astype(np.int64) ** 2, dtype=np.int64))
            var = _mean_of_sumsq_q(s2, n_el, fmt)
            y = fmt.mul(x, fx_invsqrt(np.asarray([var + eps_q]), luts, fmt)[0])
            for wn in n.weight_names:
                y = fmt.mul(y, qm.W(wn))
        elif t == Op.ROPE.value:
            y = x.copy()
            pos_q = int(fmt.quantize(float(pos)))
            idx = np.arange(0, qm.head_dim // 2)
            ang_q = fmt.mul(np.full(idx.size, pos_q, dtype=np.int64),
                            np.asarray(qm.inv_freq, dtype=np.int64))
            c_q, s_q = fx_cos(ang_q, luts, fmt), fx_sin(ang_q, luts, fmt)
            for start in range(0, x.size, qm.head_dim):
                a = start + idx
                b = a + qm.head_dim // 2
                for j in range(idx.size):
                    va, vb = fx_rope_pair_safe(qm.fmt, int(x[a[j]]), int(x[b[j]]),
                                               int(c_q[j]), int(s_q[j]))
                    y[a[j]], y[b[j]] = va, vb
        elif t == Op.ATTENTION.value:
            # 单步 decode：KV 长度=1，softmax 单元素恒为 1 -> 输出 v
            vsrc = next((i for i in n.inputs[2:] if i in vals), ins[-1])
            y = vals[vsrc].copy()
        elif t == Op.SILU.value:
            y = fx_silu(x, luts, fmt)
        elif t == Op.GELU.value:
            y = fx_gelu(x, luts, fmt)
        elif t == Op.SOFTMAX.value:
            e = fx_exp(x - int(x.max()), luts, fmt)
            tot = int(np.sum(e.astype(np.int64), dtype=np.int64))
            y = fmt.mul(e, fx_recip(np.asarray([tot]), luts, fmt)[0])
        elif t == Op.RELU.value:
            y = np.maximum(x, 0)
        elif t in (Op.ADD.value, Op.MUL.value, Op.SUB.value):
            if len(ins) < 2:
                y = x
            else:
                a, b = vals[ins[0]], vals[ins[1]]
                if t == Op.ADD.value:
                    y = fmt.saturate(a + b)
                elif t == Op.MUL.value:
                    y = fmt.mul(a, b)
                else:
                    y = fmt.saturate(a - b)
        else:
            continue

        y = np.asarray(y, dtype=np.int64)
        vals[o] = y
        trace.append((n.name, t, y.copy()))

    return trace


def fx_rope_pair_safe(fmt: FxFormat, xa: int, xb: int, c_q: int, s_q: int):
    from .fx_math import fx_rope_pair
    return fx_rope_pair(xa, xb, 0, c_q, s_q, fmt)


def fx_decode_step(qm: QuantizedModel, token: int, pos: int) -> np.ndarray:
    """定点 logits（Q.FR 整数数组）。"""
    if not qm.ir.outputs:
        raise FxRefError("GraphIR 未声明 outputs")
    trace = fx_decode_trace(qm, token, pos)
    want = qm.ir.outputs[0]
    for name, _t, val in reversed(trace):
        last = name
        for n in qm.ir.nodes:
            if n.name == name and n.outputs and n.outputs[0] == want:
                return val
    raise FxRefError(f"未找到输出张量 {want} 的计算节点（最后节点 {last if trace else None}）")
