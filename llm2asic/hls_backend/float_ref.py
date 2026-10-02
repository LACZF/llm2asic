# llm2asic/hls_backend/float_ref.py
"""GraphIR 的浮点单步 decode 参考实现。

用途：在没有 HLS 工具链的环境里校验生成的 HLS C 内核 / ONNX 导出。
`build_hls(verify=True)` 会编译生成的 C 内核并与本模块逐元素比对。

注意：本模块**不做量化**，是浮点参考。逐位一致的整数参考在
`llm2asic.rtl_backend.reference.IntModel`（现有原生 RTL 路径使用）。

语义约定（与 c_kernel.py / onnx_export.py 保持一致）：
- 词嵌入与位置嵌入是两个独立张量，相加由 `add` 节点完成；
- 单步 decode 下 KV 长度为 1，注意力退化为 `v` 透传；
- RoPE 使用序列下标 `pos`。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..ir.graph import GraphIR, Node
from ..ir.ops import Op

__all__ = ["op_type", "ref_decode_step", "ref_decode_trace"]

# 直接透传 / 由顶层内联处理的算子
_PASSTHROUGH = {
    Op.CLONE.value, Op.COPY.value, Op.CONSTANT.value, Op.RESHAPE.value,
    Op.PERMUTE.value, Op.TRANSPOSE.value, Op.KV_STORE.value, Op.KV_LOAD.value,
    Op.CONCAT.value, Op.UNFLATTEN.value, Op.GEMM.value, Op.GEMV.value,
    Op.MATMUL.value, Op.BMM.value,
}


def op_type(node: Node) -> str:
    """节点的算子名（统一成字符串）。"""
    ot = node.op_type
    return ot.value if isinstance(ot, Op) else str(ot)


def _evaluate(ir: GraphIR, token: int, pos: int):
    """核心：按拓扑序求值单步 decode，返回 `(张量表, 逐算子 trace)`。"""
    cfg = ir.config or {}
    eps = float(cfg.get("norm_eps", 1e-5) or 1e-5)
    head_dim = int(cfg.get("head_dim", 0) or 0)
    theta = float(cfg.get("rope_theta", 10000.0) or 10000.0)

    def W(wn: str) -> np.ndarray:
        w = ir.weights.get(wn)
        if w is None or w.data is None:
            raise KeyError(f"权重 {wn} 缺失数值")
        return np.asarray(w.data, dtype=np.float64)

    by_op: dict = {}
    for n in ir.nodes:
        by_op.setdefault(op_type(n), []).append(n)
    if Op.EMBEDDING.value not in by_op:
        raise ValueError("GraphIR 缺少 EMBEDDING 节点")

    emb = by_op[Op.EMBEDDING.value]
    # 第二个 EMBEDDING（若存在）是 GPT-2 风格的位置嵌入；Llama 用 RoPE。
    prologue = emb[:2] if len(emb) >= 2 else emb[:1]

    vals: dict = {}
    # 词嵌入与位置嵌入是**两个独立张量**，相加由 add 节点完成
    vals[emb[0].outputs[0]] = W(emb[0].weight_names[0])[int(token)].copy()
    if len(prologue) >= 2:
        vals[prologue[1].outputs[0]] = \
            W(prologue[1].weight_names[0])[int(pos)].copy()

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

        if t == Op.LINEAR.value:
            # IR 权重布局 [c_out, c_in]
            y = W(n.weight_names[0]) @ x
            for wn in n.weight_names[1:]:
                y = y + W(wn)
        elif t == Op.LAYERNORM.value:
            mu = x.mean()
            var = ((x - mu) ** 2).mean()
            y = (x - mu) / np.sqrt(var + eps) * W(n.weight_names[0])
            for wn in n.weight_names[1:]:
                y = y + W(wn)
        elif t == Op.RMSNORM.value:
            inv = 1.0 / np.sqrt((x ** 2).mean() + eps)
            y = x * inv
            for wn in n.weight_names:
                y = y * W(wn)
        elif t == Op.ROPE.value:
            y = x.copy()
            if head_dim:
                idx = np.arange(0, head_dim // 2)
                ang = pos / (theta ** (2.0 * idx / head_dim))
                c, s = np.cos(ang), np.sin(ang)
                for start in range(0, x.size, head_dim):
                    a = start + idx
                    b = a + head_dim // 2
                    xa, xb = x[a].copy(), x[b].copy()
                    y[a] = xa * c - xb * s
                    y[b] = xa * s + xb * c
        elif t == Op.ATTENTION.value:
            # 单步 decode：KV 长度=1，softmax 单元素恒为 1 -> 输出 v
            vsrc = next((i for i in n.inputs[2:] if i in vals), ins[-1])
            y = vals[vsrc].copy()
        elif t == Op.SILU.value:
            y = x / (1.0 + np.exp(-x))
        elif t == Op.GELU.value:
            u = 0.7978845608028654 * (x + 0.044715 * x ** 3)
            y = 0.5 * x * (1.0 + np.tanh(u))
        elif t == Op.SOFTMAX.value:
            e = np.exp(x - x.max())
            y = e / e.sum()
        elif t == Op.RELU.value:
            y = np.maximum(x, 0.0)
        elif t in (Op.ADD.value, Op.MUL.value, Op.SUB.value):
            sym = {Op.ADD.value: np.add, Op.MUL.value: np.multiply,
                   Op.SUB.value: np.subtract}[t]
            y = sym(vals[ins[0]], vals[ins[1]]) if len(ins) > 1 \
                else vals[ins[0]]
        else:
            continue

        vals[o] = np.asarray(y, dtype=np.float64)
        trace.append((n.name, t, vals[o].copy()))

    return vals, trace


def ref_decode_trace(ir: GraphIR, token: int, pos: int) -> list:
    """返回 `[(节点名, 算子, 输出数组)]`，即单步 decode 的逐算子结果。"""
    return _evaluate(ir, token, pos)[1]


def ref_decode_step(ir: GraphIR, token: int, pos: int) -> np.ndarray:
    """单步 decode 的浮点 logits 参考。"""
    if not ir.outputs:
        raise ValueError("GraphIR 未声明 outputs")
    vals, _trace = _evaluate(ir, token, pos)
    out_name = ir.outputs[0]
    if out_name not in vals:
        raise ValueError(f"未能计算输出张量 {out_name}")
    return vals[out_name]
