# llm2asic/rtl_backend/reference.py
"""整数定点推理参考实现 —— 生成黄金测试向量，也是 RTL 必须逐位吻合的唯一数值真值。

激活格式：每个张量 `T.q`（有符号整数）满足 `T.true = T.q / 2^F`（F=全局分数位）。

本文件描述的每个算子都会由 RTL 的对应引擎精确复刻（相同的整数运算与相同的 LUT），
因此 RTL 输出与 `run_decode_step()` 输出逐位一致。
"""

from __future__ import annotations

import math

import numpy as np

from . import numeric
from .numeric import F, REQUANT_S, QWeight, LUTSet, banker_round_shift

EF = 10        # softmax exp 的分数位
RR = 2 * F     # 倒数 LUT 的分数位


def clamp_act(x):
    """把中间激活裁剪到 ACT_BITS 有符号范围（与 RTL ACT 位总线语义一致）。

    位宽动态跟随 numeric.ACT_BITS（默认 16，llama_tiny 用 24）。
    """
    lo = -(1 << (numeric.ACT_BITS - 1))
    hi = (1 << (numeric.ACT_BITS - 1)) - 1
    return np.clip(np.asarray(x, dtype=np.int64), lo, hi).astype(np.int64)


def rshift_round(v: np.ndarray, n: int):
    """有符号舍入右移（half-up）。"""
    pos = (v + (1 << (n - 1))) >> n
    neg = -(((-v) + (1 << (n - 1))) >> n)
    return np.where(v >= 0, pos, neg)


class IntModel:
    """持有量化权重 + LUT，可对 GraphIR 执行解码式整数推理。"""

    def __init__(self, qweights: dict, luts: LUTSet, config: dict):
        self.qw = qweights
        self.luts = luts
        self.cfg = config
        self.hidden = config["hidden"]
        self.heads = config["num_heads"]
        self.head_dim = config["head_dim"]
        self.layers = config["num_layers"]
        self.vocab = config["vocab_size"]
        self.scale = 1.0 / (self.head_dim ** 0.5)
        self.max_seq = config["max_seq_len"]
        self.arch = str(config.get("architecture", "llama")).lower()
        self.n_inner = int(config.get("n_inner", 4 * self.hidden))
        self.kv = {}
        # softmax 分数量化：score_e = rint(dot * scale * 2^(EF-2F))
        # scale 为 2 的幂时用精确移位；否则记录为 None（v1 仅支持 2 的幂）。
        self._score_shift = None
        lg = math.log2(self.scale) if self.scale > 0 else None
        if lg is not None and abs(lg - round(lg)) < 1e-9:
            self._score_shift = int(2 * F - EF - round(lg))

    # ------------------------------------------------------------------
    # 单算子（与 RTL 一对一）
    # ------------------------------------------------------------------
    def embed(self, token: int) -> np.ndarray:
        return clamp_act(self.qw["wte_q"][token])

    def embed_gpt2(self, token: int, pos: int) -> np.ndarray:
        return clamp_act(self.qw["wte_q"][token]
                         + self.qw["wpe_q"][pos])

    def linear(self, x_q, qw: QWeight) -> np.ndarray:
        # out.q[o] = rshift_round( Σ_j wq[o,j]*x.q[j] * num[o], REQUANT_S ) (+ bias)
        acc = qw.wq.astype(np.int64) @ x_q.astype(np.int64)   # [C_out]
        num = qw.scale_num.astype(np.int64)
        out = rshift_round(acc * num, REQUANT_S)
        if getattr(qw, "bias_q", None) is not None:
            out = out + np.asarray(qw.bias_q, dtype=np.int64)
        return clamp_act(out)

    def rmsnorm(self, x_q, gamma_q: np.ndarray) -> np.ndarray:
        # out.q = rshift_round( x.q * gamma_q * c, F )，c = rsqrt_lut[mean2]
        n = x_q.shape[0]
        x64 = x_q.astype(np.int64)
        sum2 = int(np.sum(x64 * x64))
        mean2 = (sum2 + n // 2) // max(1, n) + 2          # +2 近似 eps
        idx = int(np.clip(mean2, 1, self.luts.rsqrt.shape[0] - 1))
        c = int(self.luts.rsqrt[idx])
        gq = gamma_q.astype(np.int64)
        return clamp_act(rshift_round(x64 * gq * c, F))

    def layernorm(self, x_q, gamma_q: np.ndarray, beta_q: np.ndarray) -> np.ndarray:
        """LayerNorm（GPT-2）：(x - mean) * rsqrt(1 + var) * gamma + beta。

        与 reference.rmsnorm 同一套定点：mean2 = (Σ x²)/n + 1（+1 近似 LayerNorm
        的 var+eps 中把 eps 折叠进 rsqrt 前加 1）。这里 GPT-2 用中心化均值，
        先减均值再取平方和。
        """
        n = x_q.shape[0]
        x64 = x_q.astype(np.int64)
        mean = (int(np.sum(x64)) + n // 2) // max(1, n)
        xc = x64 - mean
        sum2c = int(np.sum(xc * xc))
        # var+eps 的倒数索引：mean2 = Σxc²/n + 1
        mean2 = (sum2c + n // 2) // max(1, n) + 1
        idx = int(np.clip(mean2, 1, self.luts.rsqrt.shape[0] - 1))
        c = int(self.luts.rsqrt[idx])
        gq = gamma_q.astype(np.int64)
        bq = beta_q.astype(np.int64)
        v = rshift_round(xc * gq * c, F) + bq
        return clamp_act(v)

    def gelu(self, x_q) -> np.ndarray:
        lut = self.luts.gelu
        xb = self.luts.gelu_input_bits
        lo, hi = -2 ** (xb - 1), 2 ** (xb - 1) - 1
        xi = np.clip(x_q, lo, hi).astype(np.int64) - lo
        return clamp_act(lut[xi].astype(np.int64))

    def silu(self, x_q) -> np.ndarray:
        lut = self.luts.silu_gate
        xb = self.luts.silu_input_bits
        lo, hi = -2 ** (xb - 1), 2 ** (xb - 1) - 1
        xi = np.clip(x_q, lo, hi).astype(np.int64) - lo
        return clamp_act(lut[xi].astype(np.int64))

    def mul(self, a_q, b_q) -> np.ndarray:
        return clamp_act(rshift_round(a_q.astype(np.int64) * b_q.astype(np.int64), F))

    def add(self, a_q, b_q) -> np.ndarray:
        return clamp_act(a_q.astype(np.int64) + b_q.astype(np.int64))

    def rope(self, x_q, pos: int) -> np.ndarray:
        hidden = x_q.shape[0]
        hd = self.head_dim
        theta = float(self.cfg.get("rope_theta", 10000.0))
        out = np.zeros_like(x_q, dtype=np.int64)
        for start in range(0, hidden, hd):
            for i in range(0, hd // 2):
                a, b = i, i + hd // 2
                ang = float(pos) / (theta ** (2.0 * i / hd))
                cq = int(round(np.cos(ang) * (2.0 ** F)))
                sq = int(round(np.sin(ang) * (2.0 ** F)))
                xa, xb = int(x_q[start + a]), int(x_q[start + b])
                r1 = ((xa * cq - xb * sq) + (1 << (F - 1))) >> F
                r2 = ((xa * sq + xb * cq) + (1 << (F - 1))) >> F
                out[start + a] = r1
                out[start + b] = r2
        return clamp_act(out)

    def attention(self, qm, km, vm) -> np.ndarray:
        """单 token 多头注意。qm:[heads,hd], km/vm:[heads,seq,hd]，返回 [hidden] int。"""
        heads, hd = qm.shape
        out = np.zeros(heads * hd, dtype=np.int64)
        expneg = self.luts.exp_neg
        recip = self.luts.recip2
        for h in range(heads):
            qh = qm[h].astype(np.int64)
            kh = km[h].astype(np.int64)          # [seq, hd]
            vh = vm[h].astype(np.int64)          # [seq, hd]
            seq = kh.shape[0]
            dot = kh @ qh                         # [seq] int = real_dot * 2^{2F}
            if self._score_shift is not None:
                score_e = np.array(
                    [banker_round_shift(int(d), self._score_shift) for d in dot],
                    dtype=np.int64)
            else:
                score_true = dot.astype(np.float64) / (2.0 ** (2 * F)) * self.scale
                score_e = np.rint(score_true * (2.0 ** EF)).astype(np.int64)
            m_e = int(score_e.max()) if seq else 0
            d = np.clip(m_e - score_e, 0, expneg.shape[0] - 1).astype(np.int64)
            e = expneg[d].astype(np.int64)        # exp 整数（scale 2^EF）
            S = int(e.sum()) + 1                   # +1 防除零
            Sidx = int(np.clip(S, 1, recip.shape[0] - 1))
            rcp = int(recip[Sidx])                 # round(2^RR / S)
            num = (e[:, None] * vh).sum(axis=0)    # [hd] int64
            out[h * hd:(h + 1) * hd] = rshift_round(num * rcp, RR)
        return clamp_act(out)

    # ------------------------------------------------------------------
    # 解码一步
    # ------------------------------------------------------------------
    def run_decode_step(self, token: int, pos: int) -> np.ndarray:
        if self.arch == "gpt2":
            return self.run_decode_step_gpt2(token, pos)
        return self.run_decode_step_llama(token, pos)

    def run_decode_step_gpt2(self, token: int, pos: int) -> np.ndarray:
        h = self.embed_gpt2(token, pos)
        hidden = self.hidden
        for L in range(self.layers):
            n1 = self.layernorm(h, self.qw[f"layers.{L}.ln_1.gamma"],
                                self.qw[f"layers.{L}.ln_1.beta"])
            q = self.linear(n1, self.qw[f"layers.{L}.q"])
            k = self.linear(n1, self.qw[f"layers.{L}.k"])
            v = self.linear(n1, self.qw[f"layers.{L}.v"])
            kv = self.kv.get(L, (np.zeros((0, hidden), np.int64),
                                 np.zeros((0, hidden), np.int64)))
            kk, vv = kv
            kk = np.vstack([kk, k.reshape(1, -1)])
            vv = np.vstack([vv, v.reshape(1, -1)])
            self.kv[L] = (kk, vv)
            qm = q.reshape(self.heads, self.head_dim)
            km = kk.reshape(-1, self.heads, self.head_dim).transpose(1, 0, 2)
            vm = vv.reshape(-1, self.heads, self.head_dim).transpose(1, 0, 2)
            att = self.attention(qm, km, vm)
            o = self.linear(att, self.qw[f"layers.{L}.o"])
            h1 = self.add(h, o)
            n2 = self.layernorm(h1, self.qw[f"layers.{L}.ln_2.gamma"],
                                self.qw[f"layers.{L}.ln_2.beta"])
            fc = self.linear(n2, self.qw[f"layers.{L}.fc"])
            act = self.gelu(fc)
            proj = self.linear(act, self.qw[f"layers.{L}.proj"])
            h = self.add(h1, proj)
        nf = self.layernorm(h, self.qw["final_norm.gamma"], self.qw["final_norm.beta"])
        return self.linear(nf, self.qw["output_proj"])

    def run_decode_step_llama(self, token: int, pos: int) -> np.ndarray:
        h = self.embed(token)
        hidden = self.hidden
        for L in range(self.layers):
            n1 = self.rmsnorm(h, self.qw[f"layers.{L}.input_layernorm.gamma"])
            q = self.linear(n1, self.qw[f"layers.{L}.q"])
            k = self.linear(n1, self.qw[f"layers.{L}.k"])
            v = self.linear(n1, self.qw[f"layers.{L}.v"])
            qr = self.rope(q, pos)
            kr = self.rope(k, pos)
            kv = self.kv.get(L, (np.zeros((0, hidden), np.int64),
                                 np.zeros((0, hidden), np.int64)))
            kk, vv = kv
            kk = np.vstack([kk, kr.reshape(1, -1)])
            vv = np.vstack([vv, v.reshape(1, -1)])
            self.kv[L] = (kk, vv)
            qm = qr.reshape(self.heads, self.head_dim)
            km = kk.reshape(-1, self.heads, self.head_dim).transpose(1, 0, 2)
            vm = vv.reshape(-1, self.heads, self.head_dim).transpose(1, 0, 2)
            att = self.attention(qm, km, vm)
            h1 = self.add(h, att)
            n2 = self.rmsnorm(h1, self.qw[f"layers.{L}.post_attention_layernorm.gamma"])
            gg = self.linear(n2, self.qw[f"layers.{L}.g"])
            uu = self.linear(n2, self.qw[f"layers.{L}.u"])
            us = self.silu(uu)
            mm = self.mul(gg, us)
            dd = self.linear(mm, self.qw[f"layers.{L}.d"])
            h = self.add(h1, dd)
        nf = self.rmsnorm(h, self.qw["final_norm.gamma"])
        return self.linear(nf, self.qw["output_proj"])
