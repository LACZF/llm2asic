# llm2asic/quantizer/rom.py
"""权重 ROM / LUT 初始化文件生成（.mem，供 $readmemh）。

对应设计文档 quantizer.md §6（Step E）。
"""

from __future__ import annotations

import os

import numpy as np


def _hex_line(v: int, width_bits: int) -> str:
    nhex = max(1, (width_bits + 3) // 4)
    return f"{int(v) & ((1 << width_bits) - 1):0{nhex}x}"


def write_weight_rom(words, word_width_bits: int, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "w") as f:
        for w in words:
            f.write(_hex_line(w, word_width_bits) + "\n")


def write_scale_rom(num: np.ndarray, bits: int, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "w") as f:
        for v in num:
            f.write(f"{int(v) & ((1 << bits) - 1):0{max(1,(bits+3)//4)}x}\n")


def write_lut_rom(values: np.ndarray, bits: int, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "w") as f:
        for v in values:
            # 有符号值转二进制补码
            u = int(v) & ((1 << bits) - 1)
            f.write(f"{u:0{max(1,(bits+3)//4)}x}\n")


def write_embed_rom(wte_q: np.ndarray, bits: int, path: str) -> None:
    """嵌入矩阵以定点整数逐元素写入（行主序）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    flat = wte_q.reshape(-1)
    with open(path, "w") as f:
        for v in flat:
            u = int(v) & ((1 << bits) - 1)
            f.write(f"{u:0{max(1,(bits+3)//4)}x}\n")
