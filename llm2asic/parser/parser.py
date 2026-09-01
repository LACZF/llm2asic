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
    if bias_name is not None:
        g.weights.setdefault(bias_name, WeightDesc(bias_name, [out_features]))
    node = Node(
        name=name, op_type=Op.LINEAR,
        inputs=[inp], outputs=[out],
        attributes={"in_features": in_features, "out_features": out_features},
        weight_names=[weight_name, bias_name] if bias_name else [weight_name],
    )
    g.nodes.append(node)
    return out


def _layernorm(g: GraphIR, name: str, inp: str, wname: str, bname: str,
               hidden: int, eps: float) -> str:
    """LayerNorm（GPT-2）：(x - mean)/sqrt(var+eps) * gamma + beta。"""
    out = f"{name}_out"
    g.weights.setdefault(wname, WeightDesc(wname, [hidden]))
    g.weights.setdefault(bname, WeightDesc(bname, [hidden]))
    g.nodes.append(Node(
        name=name, op_type=Op.LAYERNORM, inputs=[inp], outputs=[out],
        attributes={"normalized_shape": [hidden], "eps": eps},
        weight_names=[wname, bname],
    ))
    return out


def _gelu(g: GraphIR, name: str, inp: str) -> str:
    out = f"{name}_out"
    g.nodes.append(Node(name=name, op_type=Op.GELU, inputs=[inp], outputs=[out],
                        attributes={}, weight_names=[]))
    return out


def _pos_add(g: GraphIR, name: str, tok_emb: str, pos_emb: str) -> str:
    out = f"{name}_out"
    g.nodes.append(Node(name=name, op_type=Op.ADD, inputs=[tok_emb, pos_emb],
                        outputs=[out], attributes={}, weight_names=[]))
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
    arch = str(lm.config.get("architecture", lm.config.get("model_type",
              lm.config.get("arch", "llama")))).lower()
    if arch == "gpt2":
        return _build_gpt2_graph(lm)
    return _build_llama_graph(lm)


def _build_llama_graph(lm: LoadedModel) -> GraphIR:
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


def _build_gpt2_graph(lm: LoadedModel) -> GraphIR:
    """GPT-2：绝对位置嵌入 + LayerNorm + GELU + 融合 c_attn(拆 qkv)。

    canonical 权重名（由 normalize 映射）：
      wte, wpe, layers.{i}.ln_1.{weight,bias}, layers.{i}.c_attn.{weight,bias},
      layers.{i}.c_attn_o.{weight,bias}, layers.{i}.ln_2.{weight,bias},
      layers.{i}.mlp_fc.{weight,bias}, layers.{i}.mlp_proj.{weight,bias},
      final_norm.{weight,bias}, lm_head.weight
    """
    import numpy as np
    cfg = lm.config
    vocab = int(cfg.get("vocab_size", cfg.get("n_vocab", 64)))
    hidden = int(cfg.get("hidden", cfg.get("n_embd", 32)))
    num_layers = int(cfg.get("num_layers", cfg.get("n_layer", 2)))
    heads = int(cfg.get("num_heads", cfg.get("n_head", 4)))
    head_dim = int(cfg.get("head_dim", max(1, hidden // heads)))
    n_inner = int(cfg.get("n_inner", cfg.get("intermediate_size", 4 * hidden)))
    max_seq = int(cfg.get("max_seq_len", cfg.get("n_positions", 16)))
    eps = float(cfg.get("norm_eps", cfg.get("layer_norm_eps", 1e-5)))
    tied = bool(cfg.get("tied_embedding", cfg.get("tie_word_embeddings", False)))

    g = GraphIR(name=cfg.get("name", "gpt2"))
    g.config = {
        "name": cfg.get("name", "gpt2"), "architecture": "gpt2",
        "vocab_size": vocab, "hidden": hidden, "num_layers": num_layers,
        "num_heads": heads, "head_dim": head_dim,
        "max_seq_len": max_seq, "rope_theta": 10000.0,
        "norm_eps": eps, "tied_embedding": tied, "seq_dim": 0,
        "n_inner": n_inner,
    }
    g.tensors["tokens"] = TensorDesc("tokens", [1], "int32")
    g.tensors["pos"] = TensorDesc("pos", [1], "int32")
    tmp = [0]

    wte = cfg.get("wte_name", "wte")
    wpe = cfg.get("wpe_name", "wpe")
    g.weights.setdefault(wte, WeightDesc(wte, [vocab, hidden]))
    g.weights.setdefault(wpe, WeightDesc(wpe, [max_seq, hidden]))
    # 1) token 嵌入
    g.nodes.append(Node(name="embed", op_type=Op.EMBEDDING, inputs=["tokens"],
                        outputs=["h_tok"], attributes={"vocab": vocab,
                        "embedding_dim": hidden}, weight_names=[wte]))
    # 2) 位置嵌入（按序列下标查表后加到 token 嵌入）
    g.nodes.append(Node(name="posemb", op_type=Op.EMBEDDING, inputs=["pos"],
                        outputs=["h_pos"], attributes={"vocab": max_seq,
                        "embedding_dim": hidden, "is_position": True},
                        weight_names=[wpe]))
    h = _pos_add(g, "p0", "h_tok", "h_pos")

    for i in range(num_layers):
        p = f"layers.{i}."
        n1 = _layernorm(g, f"l{i}.n1", h, p + "ln_1.weight", p + "ln_1.bias", hidden, eps)
        # 融合 c_attn -> 拆为 q/k/v（权重在 _finalize_weights 中按 row 切片）
        pre = _linear(g, f"l{i}.cattn", n1, p + "c_attn.weight", tmp,
                      hidden, 3 * hidden, p + "c_attn.bias")
        q = _linear(g, f"l{i}.q", pre, p + "q.weight", tmp, hidden, hidden)
        k = _linear(g, f"l{i}.k", pre, p + "k.weight", tmp, hidden, hidden)
        v = _linear(g, f"l{i}.v", pre, p + "v.weight", tmp, hidden, hidden)
        # 解码式因果注意力（与 llama 相同的 attn 内核；GPT-2 无 RoPE，直接喂 q/k/v）
        attn = _attention(g, f"l{i}.attn", q, k, v, "pos", hidden, heads,
                          head_dim, True, 1.0 / (head_dim ** 0.5))
        o = _linear(g, f"l{i}.o", attn, p + "c_attn_o.weight", tmp,
                    hidden, hidden, p + "c_attn_o.bias")
        h1 = _add(g, f"l{i}.h1", h, o)
        g.config["layer_index"] = i
        n2 = _layernorm(g, f"l{i}.n2", h1, p + "ln_2.weight", p + "ln_2.bias", hidden, eps)
        fc = _linear(g, f"l{i}.fc", n2, p + "mlp_fc.weight", tmp, hidden, n_inner,
                     p + "mlp_fc.bias")
        act = _gelu(g, f"l{i}.gelu", fc)
        proj = _linear(g, f"l{i}.proj", act, p + "mlp_proj.weight", tmp,
                       n_inner, hidden, p + "mlp_proj.bias")
        h = _add(g, f"l{i}.h", h1, proj)

    nf = _layernorm(g, "final.n", h, "final_norm.weight", "final_norm.bias", hidden, eps)
    if tied:
        g.nodes.append(Node(name="output_proj", op_type=Op.LINEAR,
                            inputs=[nf], outputs=["logits"],
                            attributes={"in_features": hidden, "out_features": vocab,
                                        "vec_rows_as_embedding": True,
                                        "transpose_weight": True},
                            weight_names=[wte]))
    else:
        g.weights.setdefault("lm_head.weight", WeightDesc("lm_head.weight", [vocab, hidden]))
        g.nodes.append(Node(name="output_proj", op_type=Op.LINEAR,
                            inputs=[nf], outputs=["logits"],
                            attributes={"in_features": hidden, "out_features": vocab},
                            weight_names=["lm_head.weight"]))

    g.inputs = ["tokens"]
    g.outputs = ["logits"]

    # c_attn 融合权重 -> q/k/v 切片
    _split_cattn(g, lm, hidden)
    _finalize_weights(g, lm)

    from .shape import infer_shapes
    infer_shapes(g)
    return g


def _split_cattn(g: GraphIR, lm: LoadedModel, hidden: int) -> None:
    """把每层的融合 c_attn 权重按行切为 q/k/v 三个标准线性权重。"""
    import numpy as np
    for i in range(g.config["num_layers"]):
        src = f"layers.{i}.c_attn.weight"
        sbias = f"layers.{i}.c_attn.bias"
        if src in lm.weights:
            W = np.asarray(lm.weights[src], dtype=np.float32)
            B = np.asarray(lm.weights.get(sbias, np.zeros(W.shape[0])), dtype=np.float32)
            if W.shape[0] != 3 * hidden:
                raise ModelParseError(
                    f"c_attn 输出维 {W.shape[0]} != 3*hidden {3*hidden}")
            sliced = {"q": W[:hidden], "k": W[hidden:2*hidden], "v": W[2*hidden:]}
        else:
            rng = np.random.default_rng(i)
            sliced = {nm: (rng.standard_normal((hidden, hidden)) * 0.02).astype(np.float32)
                      for nm in "qkv"}
        for nm, data in sliced.items():
            k = f"layers.{i}.{nm}.weight"
            lm.weights[k] = data
        # c_attn 本体重不再作为图权重（已切片），移除以避免多余随机填充
        g.weights.pop(f"layers.{i}.c_attn.weight", None)
        g.weights.pop(f"layers.{i}.c_attn.bias", None)
        # 删除 cattn（融合预投影）节点，q/k/v 输入改接到 n1 输出
        cattn_node = None
        for n in g.nodes:
            if n.name == f"l{i}.cattn":
                cattn_node = n
                break
        if cattn_node is not None:
            source_out = cattn_node.inputs[0]      # n1 的输出
            g.nodes.remove(cattn_node)
            for nm in ("q", "k", "v"):
                for poll in g.nodes:
                    if poll.name == f"l{i}.{nm}":
                        poll.inputs = [source_out]


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


def build(lm: LoadedModel) -> GraphIR:
    return build_graph(lm)


def parse(lm: LoadedModel) -> GraphIR:
    """Parser 主入口：LoadedModel -> LLM-IR。"""
    return build_graph(lm)
