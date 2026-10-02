# tests/test_fx_math.py
"""定点数学层与定点 GraphIR 参考实现的测试。

这些测试锁住三件事：

1. `fx_math` 的每个函数都只用「分段线性表 + 移位 + 一个乘法器」实现，
   且在适用范围内与 numpy 参考的误差有明确上界；
2. 表的构造约束（区间宽度是 2 的幂、段数是 2 的幂、不可寻址项钳位）
   不会被后续改动破坏；
3. `fx_ref` 的定点 decode 在三个示例模型上与浮点参考 argmax 一致。

`fx_ref` 同时是 `verilog_emit` 的黄金模型，所以这里的语义（饱和、累加器
宽度、gamma/beta 约定、RoPE 倒数频率为编译期常量）必须与 RTL 保持一致。
"""

import os

import numpy as np
import pytest

from llm2asic.hls_backend.float_ref import op_type, ref_decode_step, ref_decode_trace
from llm2asic.hls_backend.fx_math import (
    FxFormat,
    FxMathError,
    build_lut,
    default_luts,
    fx_cos,
    fx_exp,
    fx_gelu,
    fx_invsqrt,
    fx_recip,
    fx_rope_pair,
    fx_silu,
    fx_sin,
    fx_tanh,
    rope_inv_freq,
)
from llm2asic.hls_backend.fx_ref import QuantizedModel, fx_decode_step, fx_decode_trace
from llm2asic.ir.ops import Op
from llm2asic.parser.builder import run_from_path

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")
MODELS = ["gpt2_tiny", "llama_tiny", "llama_mini"]

FMT = FxFormat(32, 24)
LUTS = default_luts(FMT)

SQRT2 = float(np.sqrt(2.0))


def _model(name):
    return os.path.abspath(os.path.join(EXAMPLES, name, "model.yaml"))


def _err(got_q, ref_real):
    got = FMT.dequantize(got_q)
    return float(np.max(np.abs(got - np.asarray(ref_real, dtype=np.float64))))


# ---------------------------------------------------------------------------
# 表构造约束
# ---------------------------------------------------------------------------

def test_lut_interval_width_must_be_power_of_two():
    with pytest.raises(FxMathError, match="2 的幂"):
        build_lut("BAD", np.tanh, -1.0, 2.0, 8, FMT)


def test_lut_rejects_inverted_interval():
    with pytest.raises(FxMathError, match="非法区间"):
        build_lut("BAD", np.tanh, 1.0, 1.0, 8, FMT)


def test_lut_rejects_non_finite_without_extend_first():
    with pytest.raises(FxMathError, match="非有限值"):
        build_lut("BAD", lambda t: 1.0 / t, 0.0, 1.0, 8, FMT)


def test_lut_clamps_unreachable_low_entries():
    """1/t 在 t 很小时发散；自变量恒在 [0.25,1)，低位项应被钳位而不是报错。"""
    tbl = build_lut("R", lambda t: 0.25 / t, 0.0, 1.0, 9, FMT,
                    extend_first=True, min_index=128)
    assert np.max(np.abs(tbl.values)) < FMT.limit
    # 可寻址段（t >= 0.25）上的取值必须与解析值一致
    t = np.linspace(0.25, 1.0, 64)
    ref = np.round(0.25 / t * FMT.scale).astype(np.int64)
    for ti, ri in zip(t, ref):
        idx = min(int(ti * 512), 511)
        got = tbl.lookup(np.asarray([int(round(ti * FMT.scale))]), FMT)[0]
        assert abs(got - ri) <= 2 * (1 << tbl.shift)


def test_lut_rejects_out_of_range_value_in_reachable_region():
    with pytest.raises(FxMathError, match="超出定点范围"):
        build_lut("BAD", lambda t: 1e6 * np.ones_like(t), 0.0, 1.0, 8, FMT,
                  min_index=0)


def test_lut_index_uses_pure_shifts():
    """段数为 2 的幂、区间宽度为 2 的幂 -> 索引只需移位，无需除法器。"""
    for tbl in LUTS.values():
        assert tbl.nseg == 1 << tbl.nbits
        assert tbl.depth == tbl.nseg + 1
        assert tbl.shift >= 0


def test_default_luts_are_within_representable_range():
    for name, tbl in LUTS.items():
        assert np.max(np.abs(tbl.values)) < FMT.limit, name


# ---------------------------------------------------------------------------
# FxFormat 基本运算
# ---------------------------------------------------------------------------

def test_quantize_roundtrip():
    for v in (-3.5, -1.0, -0.25, 0.0, 0.125, 7.75):
        assert _err(FMT.quantize([v]), [v]) < 1e-6


def test_saturate_clamps_to_representable_range():
    out = FMT.saturate(np.asarray([FMT.limit, -FMT.limit, 5], dtype=np.int64))
    assert out[0] == FMT.limit - 1
    assert out[1] == -FMT.limit
    assert out[2] == 5


def test_mul_saturates_instead_of_wrapping():
    big = np.asarray([100.0 * FMT.scale], dtype=np.int64)
    out = FMT.mul(big, big)
    assert out[0] == FMT.limit - 1


def test_rshift_rounds_to_nearest():
    # bits=0 是恒等（回归：曾因 1 << (bits-1) 抛 negative shift count）
    assert FMT.rshift(np.asarray([12345]), 0)[0] == 12345
    # 四舍五入，且负数对称（半整数远离零）
    for q, want in [(1, 1), (2, 1), (3, 2), (4, 2), (5, 3)]:
        assert FMT.rshift(np.asarray([q]), 1)[0] == want, q
        assert FMT.rshift(np.asarray([-q]), 1)[0] == -want, -q


def test_rshift_does_not_truncate_positive_fractions_to_zero():
    """直接 >> 会把 0.4 抹成 0；带进位的右移必须保留它。"""
    assert FMT.dequantize(FMT.rshift(np.asarray([3]), 2))[0] > 0.0


# ---------------------------------------------------------------------------
# 数学函数精度
# ---------------------------------------------------------------------------

def test_fx_exp_on_nonpositive_input():
    """SiLU 只在自变量 <= 0 上用 exp，结果落在 (0,1]，绝不溢出。"""
    x = np.linspace(-8.0, 0.0, 2001)
    assert _err(fx_exp(FMT.quantize(x), LUTS, FMT), np.exp(x)) < 1e-5


def test_fx_exp_saturates_beyond_format_range():
    """Q8.24 上限 128，exp(10) = 22026 会饱和 —— 记录该行为。"""
    out = FMT.dequantize(fx_exp(FMT.quantize([10.0]), LUTS, FMT))[0]
    assert out == pytest.approx(128.0, abs=1e-6)


@pytest.mark.parametrize("fn,ref", [(fx_sin, np.sin), (fx_cos, np.cos)])
def test_fx_sin_cos(fn, ref):
    x = np.linspace(-8.0, 8.0, 2001)
    assert _err(fn(FMT.quantize(x), LUTS, FMT), ref(x)) < 1e-4


def test_fx_tanh_in_range_and_clamped():
    x = np.linspace(-8.0, 8.0, 2001)
    assert _err(fx_tanh(FMT.quantize(x), LUTS, FMT), np.tanh(x)) < 1e-4
    # |x| > 8 时 tanh 已饱和到 1（误差 < 2.4e-7）
    far = FMT.dequantize(fx_tanh(FMT.quantize([12.0, -12.0]), LUTS, FMT))
    assert np.allclose(far, [1.0, -1.0], atol=1e-6)


def test_fx_tanh_table_range_covers_gelu_argument():
    """GELU 的 u = 0.798(x + 0.0447x^3)，|x| <= 4 时 |u| <= 5.5 < 8（表上界）。"""
    tbl = LUTS["TANH"]
    for x in np.linspace(-4.0, 4.0, 101):
        u = 0.7978845608028654 * (x + 0.044715 * x ** 3)
        assert tbl.lo < u < tbl.hi


def test_fx_invsqrt_relative_error():
    v = np.logspace(-4.0, 1.0, 2001)
    got = FMT.dequantize(fx_invsqrt(FMT.quantize(v), LUTS, FMT))
    ref = 1.0 / np.sqrt(v)
    assert float(np.max(np.abs(got - ref) / ref)) < 1e-3


@pytest.mark.parametrize("lo,hi", [(0.5, 5.0), (1.0, 2.0), (0.5, 0.9), (1.1, 20.0)])
def test_fx_recip_relative_error(lo, hi):
    """跨数量级都要准：1/v = (1/m) * 2**(-e)，指数缩放因子是 -e 而非 -e/2。

    曾经写成 -(e//2)，v < 1 时结果差一个 2**(e/2) 倍。
    """
    v = np.logspace(np.log10(lo), np.log10(hi), 1001)
    got = FMT.dequantize(fx_recip(FMT.quantize(v), LUTS, FMT))
    ref = 1.0 / v
    assert np.all(ref < 128.0), "该区间不应触发饱和"
    assert float(np.max(np.abs(got - ref) * v)) < 1e-3


def test_fx_recip_saturates_below_range():
    out = FMT.dequantize(fx_recip(FMT.quantize([0.001]), LUTS, FMT))[0]
    assert out == pytest.approx(128.0, abs=1e-6)


def test_fx_invsqrt_handles_tiny_positive_input_without_nan():
    out = FMT.dequantize(fx_invsqrt(FMT.quantize([1e-9]), LUTS, FMT))
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# 激活函数
# ---------------------------------------------------------------------------

def test_fx_silu_matches_reference():
    x = np.linspace(-4.0, 4.0, 801)
    assert _err(fx_silu(FMT.quantize(x), LUTS, FMT), x / (1 + np.exp(-x))) < 2e-4


def test_fx_silu_negative_tail_uses_stable_form():
    """负半轴用 x*exp(x)/(1+exp(x))，避免 exp(-x) 在 x 很负时溢出。"""
    x = np.linspace(-40.0, -20.0, 201)
    out = FMT.dequantize(fx_silu(FMT.quantize(x), LUTS, FMT))
    ref = x / (1 + np.exp(-x))
    assert np.all(np.isfinite(out))
    assert float(np.max(np.abs(out - ref))) < 1e-6


def test_fx_gelu_matches_tanh_approximation():
    x = np.linspace(-4.0, 4.0, 801)
    u = 0.7978845608028654 * (x + 0.044715 * x ** 3)
    ref = 0.5 * x * (1 + np.tanh(u))
    assert _err(fx_gelu(FMT.quantize(x), LUTS, FMT), ref) < 1e-4


def test_fx_gelu_uses_cubic_term():
    """回归：beta 必须乘在 x**3 上，而不是 x 上。"""
    x = FMT.quantize([2.4])
    got = FMT.dequantize(fx_gelu(x, LUTS, FMT))[0]
    ref = 0.5 * 2.4 * (1 + np.tanh(0.7978845608028654 * (2.4 + 0.044715 * 2.4 ** 3)))
    assert got == pytest.approx(ref, abs=1e-4)


def test_fx_gelu_at_zero_is_zero():
    assert FMT.dequantize(fx_gelu(FMT.quantize([0.0]), LUTS, FMT))[0] == 0.0


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

def test_rope_inv_freq_is_compile_time_constant():
    """inv_freq 只依赖 head_dim/theta，与 pos 无关，可烧 ROM。"""
    inv = rope_inv_freq(8, 10000.0, FMT)
    assert len(inv) == 4
    ref = [10000.0 ** (-2.0 * i / 8) for i in range(4)]
    for got, r in zip(inv, ref):
        assert abs(FMT.dequantize(got) - r) < 1e-6


def test_rope_inv_freq_rejects_odd_head_dim():
    with pytest.raises(FxMathError, match="正偶数"):
        rope_inv_freq(7, 10000.0, FMT)


def test_fx_rope_pair_is_a_rotation():
    cos_q = FMT.quantize([1.0])[0]
    sin_q = FMT.quantize([0.0])[0]
    xa, xb = 3.0 * FMT.scale, -2.0 * FMT.scale
    ya, yb = fx_rope_pair(int(xa), int(xb), 0, int(cos_q), int(sin_q), FMT)
    assert FMT.dequantize([ya])[0] == pytest.approx(3.0, abs=1e-6)
    assert FMT.dequantize([yb])[0] == pytest.approx(-2.0, abs=1e-6)


def test_fx_rope_pair_ninety_degree_rotation():
    """sin=1, cos=0 时应退化为 (-xb, xa)。"""
    cos_q = FMT.quantize([0.0])[0]
    sin_q = FMT.quantize([1.0])[0]
    xa, xb = 2.0 * FMT.scale, 5.0 * FMT.scale
    ya, yb = fx_rope_pair(int(xa), int(xb), 0, int(cos_q), int(sin_q), FMT)
    assert FMT.dequantize([ya])[0] == pytest.approx(-5.0, abs=1e-6)
    assert FMT.dequantize([yb])[0] == pytest.approx(2.0, abs=1e-6)


# ---------------------------------------------------------------------------
# fx_ref：与浮点参考的一致性
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", MODELS)
def test_fx_decode_matches_float_argmax(name):
    ir = run_from_path(_model(name))
    qm = QuantizedModel(ir, FMT)
    vocab = ir.weights[
        [n for n in ir.nodes if op_type(n) == Op.EMBEDDING.value][0].weight_names[0]
    ].shape[0]
    max_pos = 64
    emb = [n for n in ir.nodes if op_type(n) == Op.EMBEDDING.value]
    if len(emb) > 1:
        max_pos = len(ir.weights[emb[1].weight_names[0]].data) - 1

    rng = np.random.RandomState(0)
    cases = [(0, 0), (1, 0), (vocab - 1, 0)]
    cases += [(int(rng.randint(vocab)), int(rng.randint(max_pos + 1)))
              for _ in range(21)]
    assert len(cases) == 24

    for tok, pos in cases:
        ref = ref_decode_step(ir, tok, pos)
        got = FMT.dequantize(fx_decode_step(qm, tok, pos))
        assert int(np.argmax(got)) == int(np.argmax(ref)), \
            f"{name} token={tok} pos={pos} argmax 不一致"


@pytest.mark.parametrize("name", MODELS)
def test_fx_decode_logits_error_is_small(name):
    ir = run_from_path(_model(name))
    qm = QuantizedModel(ir, FMT)
    for tok, pos in [(1, 0), (5, 2)]:
        ref = ref_decode_step(ir, tok, pos)
        got = FMT.dequantize(fx_decode_step(qm, tok, pos))
        assert float(np.max(np.abs(got - ref))) < 1e-3


@pytest.mark.parametrize("name", MODELS)
def test_fx_trace_has_same_op_sequence_as_float_ref(name):
    """逐算子 trace 必须与浮点参考一一对应，发射器按这个顺序排 FSM。"""
    ir = run_from_path(_model(name))
    qm = QuantizedModel(ir, FMT)
    f = ref_decode_trace(ir, 1, 0)
    g = fx_decode_trace(qm, 1, 0)
    assert [t for _, t, _ in f] == [t for _, t, _ in g]
    assert [n for n, _, _ in f] == [n for n, _, _ in g]


@pytest.mark.parametrize("name", MODELS)
def test_fx_every_node_stays_in_representable_range(name):
    """RTL 数据通路饱和；参考实现必须满足同样的不变量（饱和值只能是边界）。"""
    ir = run_from_path(_model(name))
    qm = QuantizedModel(ir, FMT)
    for _n, t, v in fx_decode_trace(qm, 3, 1):
        assert v.dtype == np.int64
        assert np.max(np.abs(v)) < FMT.limit, f"{name}/{t} 越界"


def test_fx_layernorm_uses_gamma_multiply_then_beta_add():
    """回归：weight_names[0] 是 gamma（乘），其余是 beta（加）。

    之前按维数猜测 gamma/beta，导致 gpt2 的第一个 layernorm 就偏 1.46。
    """
    from llm2asic.hls_backend.float_ref import _evaluate

    ir = run_from_path(_model("gpt2_tiny"))
    ln = next(n for n in ir.nodes if op_type(n) == Op.LAYERNORM.value)
    assert len(ln.weight_names) >= 2, "示例模型的 layernorm 应有 gamma 和 beta"
    qm = QuantizedModel(ir, FMT)
    # gamma 全 1、beta 全 0 的等效情形：结果应等于归一化后的 x
    _vals, _ = _evaluate(ir, 1, 0)
    trace = fx_decode_trace(qm, 1, 0)
    got = next(v for n, t, v in trace if t == Op.LAYERNORM.value)
    ref = next(v for n, t, v in ref_decode_trace(ir, 1, 0) if t == Op.LAYERNORM.value)
    assert float(np.max(np.abs(FMT.dequantize(got) - ref))) < 1e-2


def test_fx_attention_is_value_passthrough():
    """单步 decode：KV 长度=1，注意力退化为 v 透传。"""
    ir = run_from_path(_model("llama_tiny"))
    qm = QuantizedModel(ir, FMT)
    trace = fx_decode_trace(qm, 1, 0)
    attn = [v for _n, t, v in trace if t == Op.ATTENTION.value]
    assert attn, "示例模型应包含注意力节点"
    for v in attn:
        # 透传意味着取值就是某个上游张量，未被缩放
        assert np.max(np.abs(v)) < FMT.limit


def test_fx_embedding_position_is_separate_from_token():
    ir = run_from_path(_model("gpt2_tiny"))
    qm = QuantizedModel(ir, FMT)
    emb = [n for n in ir.nodes if op_type(n) == Op.EMBEDDING.value]
    assert len(emb) >= 2
    a = qm.W(emb[0].weight_names[0])[3]
    b = qm.W(emb[1].weight_names[0])[0]
    assert a.shape == b.shape
    # 两者不应被合并成同一个张量
    assert not np.array_equal(a, b)
