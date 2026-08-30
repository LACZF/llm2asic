# tests/test_numeric.py
"""数值单元测试：取整、LUT、量化 —— RTL 与黄金参考共同依赖的定点规则。"""

import numpy as np
import pytest

from llm2asic.rtl_backend.numeric import (
    F,
    ACT_BITS,
    REQUANT_S,
    banker_round_shift,
    gen_luts,
    quantize_weight,
)
from llm2asic.rtl_backend.reference import rshift_round


def _bnk(v, k):
    """参考实现（round-half-even）。"""
    q = v >> k
    rem = v - (q << k)
    half = 1 << (k - 1)
    if rem > half:
        return q + 1
    if rem == half:
        return q if (q % 2 == 0) else q + 1
    return q


@pytest.mark.parametrize("k", [1, 4, 8, 12, 15])
def test_banker_one_uses_reference(k):
    """banker_round_shift 与参考 round-half-even 完全一致。"""
    rng = np.random.default_rng(0)
    for _ in range(2000):
        v = int(rng.integers(-(1 << 30), 1 << 30))
        assert banker_round_shift(v, k) == _bnk(v, k)


def test_banker_round_half_even():
    """精确半步归偶。"""
    assert banker_round_shift(3, 1) == 2     # 1.5 -> 2
    assert banker_round_shift(5, 2) == 1     # 1.25 -> 1
    assert banker_round_shift(6, 2) == 2     # 1.5 -> 2
    assert banker_round_shift(2, 1) == 1     # 1.0 -> 1
    assert banker_round_shift(7, 1) == 4     # 3.5 -> 4
    assert banker_round_shift(9, 1) == 4     # 4.5 -> 4


def test_rshift_round_matches_banker_for_shift():
    """rshift_round（定点，半上调）与 banker（round-half-even）在 F 位固定时应
    由调用方选择；这里校验 rshift_round 对正负都做半上调（RTL rr 语义）。"""
    assert rshift_round(np.array([5]), 2)[0] == 1      # 1.25 -> 1
    assert rshift_round(np.array([7]), 2)[0] == 2      # 1.75 -> 2
    assert rshift_round(np.array([-7]), 2)[0] == -2    # -1.75 -> -2（半上调）
    assert rshift_round(np.array([-5]), 2)[0] == -1


def test_lut_shapes_and_nonneg():
    """LUT 尺寸与值域契约。"""
    lut = gen_luts()
    assert lut.rsqrt.shape[0] == 2**20 + 1
    assert lut.rsqrt[0] == 0
    assert np.all(lut.rsqrt[1:] > 0)
    assert lut.exp_neg.shape[0] == 2**12
    assert lut.exp_neg[0] == np.rint(2.0**F)          # exp(0)=1
    assert lut.recip2.shape[0] == 2**15 - 1   # x 从 1..2^15-1
    assert lut.recip2[0] > 0
    assert lut.sigmoid_input_bits == 10
    assert lut.silu_input_bits == 10


def test_quantize_weight_dims_and_bounds():
    """逐行对称量化保持形状与值域。"""
    rng = np.random.default_rng(1)
    w = rng.standard_normal((8, 16))
    qw = quantize_weight(w, 4, 16, name="w")
    assert qw.wq.shape == (8, 16)
    assert qw.wq.dtype == np.int64
    assert qw.wq.max() <= (1 << 3) - 1
    assert qw.wq.min() >= -(1 << 3)
    assert qw.scale_num.shape == (8,)
    assert np.all(qw.scale_num >= 1)
    assert qw.requant_shift == REQUANT_S
    assert qw.c_out == 8 and qw.c_in == 16


def test_quantize_weight_rejects_1d():
    with pytest.raises(ValueError):
        quantize_weight(np.zeros(8), 4, 16, name="bad")
