# llm2asic/quantizer/reorder.py
"""权重重排：把量化后的权重排布为 RTL GEMV 引擎可直接顺序读取的 ROM 布局。

对应设计文档 quantizer.md §4（Step C）。布局约定（RTL 与本书一致，跨组件契约）：
  - 每个输出行 o 的权重按 SIMD 个一组打包；word = SIMD 个 WW 位有符号权重。
  - ROM 地址 = o * word_per_row + (c_in_chunk)。
"""

from __future__ import annotations

import numpy as np

from ..rtl_backend.numeric import QWeight


def reorder_weight(qw: QWeight, pe: int = 1, simd: int = 8) -> tuple:
    """返回 (words[uint64], word_width_bits, simd, ww)。

    words 长度 = c_out * ceil(c_in / simd)，每个元素为 packed word。
    """
    wq = qw.wq.astype(np.int64)
    ww = qw.bit_width
    c_out, c_in = wq.shape
    chunk = max(1, simd)
    mask = (1 << ww) - 1
    word_width = chunk * ww

    words = []
    for o in range(c_out):
        row = wq[o]
        # 补齐到 chunk 整数倍（补 0）
        if c_in % chunk:
            pad = chunk - (c_in % chunk)
            row = np.concatenate([row, np.zeros(pad, np.int64)])
        n_words = len(row) // chunk
        for w in range(n_words):
            vals = row[w * chunk:(w + 1) * chunk] & mask
            word = 0
            for j, v in enumerate(vals):
                word |= int(v) << (j * ww)
            words.append(word)

    return np.array(words, dtype=np.uint64), word_width, simd, ww


def reorder_scale(qw: QWeight) -> np.ndarray:
    """返回每个输出行的 requant 分子 num[C_out]。"""
    return qw.scale_num.astype(np.int64)
