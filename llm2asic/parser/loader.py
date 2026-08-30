# llm2asic/parser/loader.py
"""模型加载器：把模型描述 + 权重加载为统一的内部表示。

当前支持两种来源：
1. 声明式模型描述（YAML + NumPy .npz 权重）——框架无关，无需 torch。
2. `torch.nn.Module`（若已安装 torch/safetensors）——可选路径。

统一暴露 `LoadedModel`：
    config     dict    模型级配置
    weights    dict[str, ndarray]  权重名 -> 数值
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import yaml


@dataclass
class LoadedModel:
    config: dict = field(default_factory=dict)
    weights: dict = field(default_factory=dict)
    source_type: str = "spec"          # spec | torch | safetensors
    spec_dir: str = ""


def _load_npz(path: str) -> dict:
    with np.load(path, allow_pickle=True) as data:
        return {k: np.asarray(v) for k, v in data.items()}


def _load_from_spec(model_path: str) -> LoadedModel:
    spec_dir = os.path.dirname(os.path.abspath(model_path))
    with open(model_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    # 找到相邻权重文件
    weights_path = cfg.get("weights_file") or os.path.join(spec_dir, "weights.npz")
    if not os.path.exists(weights_path):
        # 尝试 *.npz
        import glob
        cands = glob.glob(os.path.join(spec_dir, "*.npz"))
        if not cands:
            raise FileNotFoundError(
                f"找不到权重文件 {weights_path}；请先用 gen_model.py 生成 weights.npz"
            )
        weights_path = cands[0]
    weights = _load_npz(weights_path)
    return LoadedModel(config=cfg, weights=weights, source_type="spec", spec_dir=spec_dir)


def _load_from_torch(model_path: str) -> LoadedModel:
    """可选路径：从 torch.nn.Module / safetensors 加载。需要 torch。"""
    try:
        import torch  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "加载 torch.nn.Module 需要安装 torch；本环境未安装，请改用声明式模型描述(yaml)。"
        )

    if model_path.endswith(".safetensors"):
        from safetensors import safe_open
        weights = {}
        with safe_open(model_path, framework="np") as f:
            for k in f.keys():
                weights[k] = f.get_tensor(k)
        return LoadedModel(config={}, weights=weights, source_type="safetensors")

    if isinstance_import(model_path):
        return LoadedModel(config={}, weights=state_dict_of(model_path), source_type="torch")

    raise ValueError(f"不支持的模型来源: {model_path}")


def isinstance_import(obj) -> bool:
    try:
        import torch
        return isinstance(obj, torch.nn.Module)
    except Exception:
        return False


def state_dict_of(obj) -> dict:
    try:
        import torch
        sd = obj.state_dict()
        return {k: v.detach().cpu().numpy() for k, v in sd.items()}
    except Exception:
        return {}


def load_model(model_path: str) -> LoadedModel:
    """统一入口：根据路径/对象类型选择加载方式。"""
    if hasattr(model_path, "state_dict"):   # 一个 torch.nn.Module 实例
        return _load_from_torch(model_path)
    if isinstance(model_path, str) and model_path.endswith((".npz", ".safetensors", ".pth", ".bin")):
        return _load_from_torch(model_path) if model_path.endswith((".safetensors", ".pth", ".bin")) \
            else LoadedModel(config={}, weights=_load_npz(model_path), source_type="spec")
    if isinstance(model_path, str) and model_path.endswith(".yaml"):
        return _load_from_spec(model_path)
    raise ValueError(f"无法识别的模型来源: {model_path!r}")
