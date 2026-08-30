# llm2asic/ir/serialize.py
"""IR JSON 序列化 / 反序列化。对应设计文档 parser.md §5。

权重数值默认外置到 `*.bin`（numpy 原始二进制），IR 中记录 `data_file`，
避免超大 JSON。加载时可选择是否回填 `data`。
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from .graph import GraphIR, Node, TensorDesc, WeightDesc


# ---------------------------------------------------------------------------
# Node / TensorDesc 序列化
# ---------------------------------------------------------------------------

def tensor_to_dict(t: TensorDesc) -> dict:
    d = {"name": t.name, "shape": list(t.shape), "dtype": t.dtype}
    if isinstance(t, WeightDesc):
        if t.bit_width is not None:
            d["bit_width"] = t.bit_width
        if t.group_size is not None:
            d["group_size"] = t.group_size
        if t.quant_scheme is not None:
            d["quant_scheme"] = t.quant_scheme
        if t.data_file is not None:
            d["data_file"] = t.data_file
        if t.data is not None and t.data.size <= 64:
            d["_data"] = t.data.tolist()
    return d


def _write_weight_bin(weight: WeightDesc, base_dir: str) -> None:
    """把权重 data 写入 base_dir 下 .bin 文件（若尚未有 data_file）。"""
    if weight.data is None:
        return
    if weight.data_file is None:
        weight.data_file = f"{weight.name}.bin"
    path = os.path.join(base_dir, weight.data_file)
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    if weight.data.dtype != np.float32:
        arr = weight.data.astype(np.float32)
    else:
        arr = weight.data
    arr.tofile(path)


def graph_to_dict(g: GraphIR, weights_dir: str | None = None) -> dict:
    """把 GraphIR 转成可 JSON 序列化的 dict。

    若 weights_dir 非 None，则把每个权重 data 外置写到该目录的 .bin 文件。
    """
    if weights_dir is not None:
        os.makedirs(weights_dir, exist_ok=True)
        for name, w in g.weights.items():
            if w.data is not None:
                _write_weight_bin(w, weights_dir)

    return {
        "name": g.name,
        "inputs": list(g.inputs),
        "outputs": list(g.outputs),
        "config": g.config,
        "nodes": [
            {
                "name": n.name,
                "op_type": n.op_type,
                "inputs": list(n.inputs),
                "outputs": list(n.outputs),
                "attributes": n.attributes,
                "weight_names": list(n.weight_names),
                "quant": n.quant,
                "source": n.source,
            }
            for n in g.nodes
        ],
        "tensors": {name: tensor_to_dict(t) for name, t in g.tensors.items()},
        "weights": {name: tensor_to_dict(w) for name, w in g.weights.items()},
        "quant_meta": _quant_meta_to_dict(g.quant_meta),
    }


def _quant_meta_to_dict(qm) -> dict:
    out = {}
    for k, v in (qm or {}).items():
        d = asdict_hard(v)
        # 去掉 numpy 不可序列化字段
        out[k] = {kk: str(vv) if isinstance(vv, np.ndarray) else vv for kk, vv in d.items()}
    return out


def asdict_hard(obj):
    from dataclasses import asdict
    return asdict(obj)


def graph_from_dict(d: dict, base_dir: str | None = None) -> GraphIR:
    g = GraphIR(
        name=d["name"],
        inputs=list(d.get("inputs", [])),
        outputs=list(d.get("outputs", [])),
        config=d.get("config", {}),
    )
    for nd in d["nodes"]:
        g.nodes.append(Node(
            name=nd["name"],
            op_type=nd["op_type"],
            inputs=list(nd.get("inputs", [])),
            outputs=list(nd.get("outputs", [])),
            attributes=nd.get("attributes", {}),
            weight_names=list(nd.get("weight_names", [])),
            quant=nd.get("quant"),
            source=nd.get("source"),
        ))
    for name, td in d["tensors"].items():
        g.tensors[name] = TensorDesc(name, td["shape"], td.get("dtype", "fp32"))
    for name, wd in d["weights"].items():
        w = WeightDesc(name, wd["shape"], wd.get("dtype", "fp32"))
        w.bit_width = wd.get("bit_width")
        w.group_size = wd.get("group_size")
        w.quant_scheme = wd.get("quant_scheme")
        w.data_file = wd.get("data_file")
        if base_dir and w.data_file and os.path.exists(os.path.join(base_dir, w.data_file)):
            w.data = np.fromfile(os.path.join(base_dir, w.data_file), dtype=np.float32)
            w.data = w.data.reshape(w.shape)
        g.weights[name] = w
    # quant_meta 保持字典原样（dict[str, dict]）
    g.quant_meta = d.get("quant_meta") or {}
    return g


def dump_ir(g: GraphIR, path: str, weights_dir: str | None = None) -> None:
    """把 GraphIR 写为 JSON。weights_dir 非 None 时将权重外置。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if weights_dir is None:
        weights_dir = os.path.join(os.path.dirname(os.path.abspath(path)), "weights")
    d = graph_to_dict(g, weights_dir=weights_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)


def load_ir(path: str, load_weights: bool = True) -> GraphIR:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    base = os.path.dirname(os.path.abspath(path))
    return graph_from_dict(d, base_dir=base)


# ---------------------------------------------------------------------------
# 通用 JSON 工具（archdesc / 报告 / 元数据）
# ---------------------------------------------------------------------------

def dump_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
