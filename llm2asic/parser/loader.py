# llm2asic/parser/loader.py
"""模型加载器：把模型描述 + 权重加载为统一的内部表示。

当前支持多种来源/格式，统一暴露 `LoadedModel`：
    config     dict    模型级配置（canonical，已推导）
    weights    dict[str, ndarray]  canonical 权重名 -> 数值
    source_type str     spec | safetensors | onnx | bin
    spec_dir   str     模型所在目录

格式：
 1. spec        声明式 YAML + .npz（框架无关，无需任何第三方）
 2. .safetensors HF 权重 + 相邻 config.json（safetensors 库，无需 torch）
 3. .onnx        ONNX 图 + 相邻 config.json（onnx 库，取 initializer 权重）
 4. .bin         PyTorch state_dict pickle（需 torch）或原始 fp32 二进制 + .yaml 形状
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import yaml

from .normalize import normalize, infer_config


@dataclass
class LoadedModel:
    config: dict = field(default_factory=dict)
    weights: dict = field(default_factory=dict)
    source_type: str = "spec"          # spec | safetensors | onnx | bin
    spec_dir: str = ""


def _load_npz(path: str) -> dict:
    with np.load(path, allow_pickle=True) as data:
        return {k: np.asarray(v) for k, v in data.items()}


def _load_adjacent_config(spec_dir: str) -> dict | None:
    """读取模型目录下 config.json，供 external 权重格式推导 canonical config。"""
    for cand in ("config.json",):
        p = os.path.join(spec_dir, cand)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


def _load_from_spec(model_path: str) -> LoadedModel:
    """YAML + 相邻 .npz（canonical 权重名，paser 直接可用）。"""
    spec_dir = os.path.dirname(os.path.abspath(model_path))
    with open(model_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    weights_path = cfg.get("weights_file") or os.path.join(spec_dir, "weights.npz")
    if not os.path.exists(weights_path):
        import glob
        cands = glob.glob(os.path.join(spec_dir, "*.npz"))
        if not cands:
            raise FileNotFoundError(
                f"找不到权重文件 {weights_path}；请先用 gen_model.py 生成 weights.npz")
        weights_path = cands[0]
    weights = _load_npz(weights_path)
    return LoadedModel(config=cfg, weights=weights, source_type="spec", spec_dir=spec_dir)


def _load_from_safetensors(model_path: str) -> LoadedModel:
    """safetensors + 相邻 config.json。safetensors 库可用 numpy 独立加载，无需 torch。"""
    try:
        from safetensors import safe_open
    except ImportError:
        raise RuntimeError(
            "加载 .safetensors 需要 safetensors 库；请 pip install safetensors")

    spec_dir = os.path.dirname(os.path.abspath(model_path))
    weights = {}
    with safe_open(model_path, framework="np") as f:
        for k in f.keys():
            weights[k] = f.get_tensor(k)
    raw_cfg = _load_adjacent_config(spec_dir) or {}
    return LoadedModel(
        config=infer_config(raw_cfg),
        weights=normalize(weights),
        source_type="safetensors", spec_dir=spec_dir)


def _load_from_onnx(model_path: str) -> LoadedModel:
    """ONNX 图：读取所有 initializer（权重）张量 + 相邻 config.json。"""
    try:
        import onnx
    except ImportError:
        raise RuntimeError(
            "加载 .onnx 需要 onnx 库；请 pip install onnx")

    spec_dir = os.path.dirname(os.path.abspath(model_path))
    model = onnx.load(model_path)
    graph = model.graph
    weights = {}
    for init in graph.initializer:
        arr = onnx.numpy_helper.to_array(init)
        weights[init.name] = arr
    raw_cfg = _load_adjacent_config(spec_dir) or {}
    return LoadedModel(
        config=infer_config(raw_cfg),
        weights=normalize(weights),
        source_type="onnx", spec_dir=spec_dir)


def _load_from_hf_bin(model_path: str) -> LoadedModel:
    """PyTorch state_dict pickle（HF .bin）。需要 torch。"""
    try:
        import torch
        sd = torch.load(model_path, map_location="cpu")
        if hasattr(sd, "state_dict"):
            sd = sd.state_dict()
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        weights = {k: v.detach().cpu().numpy() if hasattr(v, "detach")
                   else np.asarray(v) for k, v in sd.items()}
    except ImportError:
        raise RuntimeError(
            "加载 HF .bin（PyTorch pickle）需要 torch；本环境未安装。"
            "请改用 .npz / .safetensors / .onnx 格式，或 pip install torch")
    spec_dir = os.path.dirname(os.path.abspath(model_path))
    raw_cfg = _load_adjacent_config(spec_dir) or {}
    return LoadedModel(
        config=infer_config(raw_cfg),
        weights=normalize(weights),
        source_type="bin", spec_dir=spec_dir)


def _load_from_raw_bin(model_path: str) -> LoadedModel:
    """原始 fp32 二进制 + 相邻 .yaml（形状/配置）。对应 exporter 的 .bin 产物。"""
    spec_dir = os.path.dirname(os.path.abspath(model_path))
    yaml_path = os.path.join(spec_dir, "model.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(
            f"原始二进制 {model_path} 需要相邻 model.yaml（提供形状/配置）")
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    # 单文件：假定包含 numpy 的 .bin（fp32 平铺），形状取自 model.yaml 的 weights 字段
    shapes = cfg.get("weights", {})
    data = np.fromfile(model_path, dtype=np.float32)
    weights = {}
    offset = 0
    for name, shape in shapes.items():
        n = int(np.prod(shape))
        weights[name] = data[offset:offset + n].reshape(shape)
        offset += n
    cfg.pop("weights", None)
    return LoadedModel(config=cfg, weights=weights, source_type="bin", spec_dir=spec_dir)


def _load_from_torch(model_path: str) -> LoadedModel:
    """可选路径：从 torch.nn.Module / safetensors 加载（兼容旧接口）。"""
    try:
        import torch  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "加载 torch.nn.Module 需要安装 torch；本环境未安装，请改用声明式模型描述(yaml)。")

    if model_path.endswith(".safetensors"):
        return _load_from_safetensors(model_path)
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
    if isinstance(model_path, str):
        if model_path.endswith(".yaml"):
            return _load_from_spec(model_path)
        if model_path.endswith(".safetensors"):
            return _load_from_safetensors(model_path)
        if model_path.endswith(".onnx"):
            return _load_from_onnx(model_path)
        if model_path.endswith(".bin"):
            # 优先原始 fp32 二进制（无需 torch），否则 PyTorch pickle
            try:
                return _load_from_raw_bin(model_path)
            except FileNotFoundError:
                return _load_from_hf_bin(model_path)
        if model_path.endswith((".npz", ".pth")):
            return LoadedModel(config={}, weights=_load_npz(model_path), source_type="spec") \
                if model_path.endswith(".npz") else _load_from_hf_bin(model_path)
    raise ValueError(f"无法识别的模型来源: {model_path!r}")
