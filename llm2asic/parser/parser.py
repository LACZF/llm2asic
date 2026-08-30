# llm2asic/parser/parser.py
"""算子映射与 LLM-IR 计算图构建。

把声明式模型描述（或 LoadedModel）归一化为扁平 DAG（内部算子集合），
并附上静态形状。本组件不引入任何硬件概念。
"""

from __future__ import annotations

from ..ir.graph import GraphIR, Node, TensorDesc, WeightDesc
from ..ir.ops import Op

from .loader import LoadedModel


class ModelParseError(Exception):
    pass


# ---------------------------------------------------------------------------
# 构建单个 linear 节点
# ---------------------------------------------------------------------------

def _linear(g: GraphIR, name: str, inp: str, weight_name: str, tmp_id,
            in_features: int, out_features: int, bias_name: str | None = None) -> str:
    out = f"t{tmp_id[0]}_{name}_out"
    tmp_id[0] += 1
    g.weights.setdefault(weight_name, WeightDesc(weight_name, [out_features, in_features]))
    node = Node(
        name=name, op_type=Op.LINEAR,
        inputs=[inp], outputs=[out],
        attributes={"in_features": in_features, "out_features": out_features},
        weight_names=[weight_name, bias_name] if bias_name else [weight_name],
    )
    g.nodes.append(node)
    return out


def _rmsnorm(g: GraphIR, name: str, inp: str, weight_name: str, hidden: int, eps: float) -> str:
    out = f"{name}_out"
    g.weights.setdefault(weight_name, WeightDesc(weight_name, [hidden]))
    g.nodes.append(Node(
        name=name, op_type=Op.RMSNORM, inputs=[inp], outputs=[out],
        attributes={"normalized_shape": [hidden], "eps": eps},
        weight_names=[weight_name],
    ))
    return out


def _rope(g: GraphIR, name: str, inp: str, pos: str, hidden: int, head_dim: int,
          theta: float) -> str:
    out = f"{name}_out"
    g.nodes.append(Node(
        name=name, op_type=Op.ROPE, inputs=[inp, pos], outputs=[out],
        attributes={"hidden": hidden, "head_dim": head_dim, "theta": theta},
        weight_names=[],
    ))
    return out


def _attention(g: GraphIR, name: str, q, k, v, pos, hidden: int, heads: int,
               head_dim: int, causal: bool, scale: float) -> str:
    out = f"{name}_out"
    g.nodes.append(Node(
        name=name, op_type=Op.ATTENTION,
        inputs=[q, k, v, pos], outputs=[out],
        attributes={"hidden": hidden, "heads": heads, "head_dim": head_dim,
                    "causal": causal, "scale": scale,
                    "layer": g.config.get("layer_index", 0)},
        weight_names=[],
    ))
    return out


def _add(g: GraphIR, name: str, a: str, b: str) -> str:
    out = f"{name}_out"
    g.nodes.append(Node(name=name, op_type=Op.ADD, inputs=[a, b], outputs=[out],
                        attributes={}, weight_names=[]))
    return out


def _silu(g: GraphIR, name: str, inp: str) -> str:
    out = f"{name}_out"
    g.nodes.append(Node(name=name, op_type=Op.SILU, inputs=[inp], outputs=[out],
                        attributes={}, weight_names=[]))
    return out


def _mul(g: GraphIR, name: str, a: str, b: str) -> str:
    out = f"{name}_out"
    g.nodes.append(Node(name=name, op_type=Op.MUL, inputs=[a, b], outputs=[out],
                        attributes={}, weight_names=[]))
    return out


# ---------------------------------------------------------------------------
# 顶层构建
# ---------------------------------------------------------------------------

def build_graph(lm: LoadedModel) -> GraphIR:
    cfg = lm.config
    vocab = int(cfg.get("vocab_size", cfg.get("vocab", 64)))
    hidden = int(cfg.get("hidden", cfg.get("hidden_size", 32)))
    num_layers = int(cfg.get("num_layers", cfg.get("n_layers", 2)))
    heads = int(cfg.get("num_heads", cfg.get("n_heads", 4)))
    head_dim = int(cfg.get("head_dim", max(1, hidden // heads)))
    max_seq = int(cfg.get("max_seq_len", 16))
    theta = float(cfg.get("rope_theta", 10000.0))
    eps = float(cfg.get("norm_eps", 1e-5))
    tied = bool(cfg.get("tied_embedding", cfg.get("tie_word_embeddings", False)))

    g = GraphIR(name=cfg.get("name", "model"))

    # 模型级配置（尽早设置，供各构建 helper 读取）
    g.config = {
        "name": cfg.get("name", "model"),
        "vocab_size": vocab,
        "hidden": hidden,
        "num_layers": num_layers,
        "num_heads": heads,
        "head_dim": head_dim,
        "max_seq_len": max_seq,
        "rope_theta": theta,
        "norm_eps": eps,
        "tied_embedding": tied,
        "seq_dim": 0,
    }

    # 图张量（静态形状）
    g.tensors["tokens"] = TensorDesc("tokens", [1], "int32")
    g.tensors["pos"] = TensorDesc("pos", [1], "int32")

    tmp = [0]
    first = cfg.get("freeze_input_layernorm", False)

    # ---- 输入嵌入 ----
    wte = cfg.get("wte_name", "wte")
    g.weights.setdefault(wte, WeightDesc(wte, [vocab, hidden]))
    emb = g.weights[wte]
    g.nodes.append(Node(name="embed", op_type=Op.EMBEDDING, inputs=["tokens"],
                        outputs=["h_emb"],
                        attributes={"vocab": vocab, "embedding_dim": hidden},
                        weight_names=[wte]))

    h_in = "h_emb"

    # ---- 逐层 ----
    for i in range(num_layers):
        p = f"layers.{i}."
        # 归一化1
        n1 = _rmsnorm(g, f"l{i}.n1", h_in, p + "input_layernorm.weight", hidden, eps)
        # Q/K/V 投影
        q = _linear(g, f"l{i}.q", n1, p + "self_attn.q_proj.weight", tmp, hidden, hidden)
        k = _linear(g, f"l{i}.k", n1, p + "self_attn.k_proj.weight", tmp, hidden, hidden)
        v = _linear(g, f"l{i}.v", n1, p + "self_attn.v_proj.weight", tmp, hidden, hidden)
        # RoPE
        qr = _rope(g, f"l{i}.qr", q, "pos", hidden, head_dim, theta)
        kr = _rope(g, f"l{i}.kr", k, "pos", hidden, head_dim, theta)
        # Attention（含 KV store/load, causal mask）
        attn = _attention(g, f"l{i}.attn", qr, kr, v, "pos", hidden, heads,
                          head_dim, True, 1.0 / (head_dim ** 0.5))
        # 残差
        h1 = _add(g, f"l{i}.h1", h_in, attn)
        # 归一化2
        n2 = _rmsnorm(g, f"l{i}.n2", h1, p + "post_attention_layernorm.weight", hidden, eps)
        # 门控 MLP
        g_gate = _linear(g, f"l{i}.g", n2, p + "mlp.gate_proj.weight", tmp, hidden, hidden)
        u = _linear(g, f"l{i}.u", n2, p + "mlp.up_proj.weight", tmp, hidden, hidden)
        us = _silu(g, f"l{i}.us", u)
        m = _mul(g, f"l{i}.m", g_gate, us)
        d = _linear(g, f"l{i}.d", m, p + "mlp.down_proj.weight", tmp, hidden, hidden)
        h = _add(g, f"l{i}.h", h1, d)
        g.config["layer_index"] = i
        h_in = h

    # ---- 输出 ----
    nf = _rmsnorm(g, "final.n", h_in, "final_norm.weight", hidden, eps)
    # 输出投影：tied 则复用 wte，否则为独立权重
    if tied:
        g.nodes.append(Node(name="output_proj", op_type=Op.LINEAR,
                            inputs=[nf], outputs=["logits"],
                            attributes={"in_features": hidden, "out_features": vocab,
                                        "vec_rows_as_embedding": True,
                                        "transpose_weight": True},
                            weight_names=[wte]))
    else:
        wout_name = cfg.get("lm_head_name", "lm_head.weight")
        g.weights.setdefault(wout_name, WeightDesc(wout_name, [vocab, hidden]))
        g.nodes.append(Node(name="output_proj", op_type=Op.LINEAR,
                            inputs=[nf], outputs=["logits"],
                            attributes={"in_features": hidden, "out_features": vocab},
                            weight_names=[wout_name]))

    g.inputs = ["tokens"]
    g.outputs = ["logits"]

    # 权重重排/data 填充 + 形状定型
    _finalize_weights(g, lm)

    # 形状推导
    from .shape import infer_shapes
    infer_shapes(g)
    return g


def _finalize_weights(g: GraphIR, lm: LoadedModel) -> None:
    """为所有尚无 data 的权重填充数值：优先来自模型权重，否则确定性伪随机。"""
    import numpy as np
    import zlib
    for name, w in g.weights.items():
        if w.data is None:
            if name in lm.weights:
                data = np.asarray(lm.weights[name], dtype=np.float32)
            else:
                rng = np.random.default_rng(zlib.crc32(name.encode("utf-8")))
                data = (rng.standard_normal(w.shape) * 0.02).astype(np.float32)
            w.data = np.ascontiguousarray(data.astype(np.float32))
        w.shape = list(int(x) for x in w.data.shape)

        # 权重命名为 {out,in} 的线性层：统一做确定性填充（若缺失形状）
        if w.data.ndim == 2:
            pass


def build(lm: LoadedModel) -> GraphIR:
    return build_graph(lm)


def parse(lm: LoadedModel) -> GraphIR:
    """Parser 主入口：LoadedModel -> LLM-IR。"""
    return build_graph(lm)
