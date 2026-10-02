# llm2asic/hls_backend/onnx_export.py
"""GraphIR -> ONNX 导出（hls4ml 前端 / 通用 ONNX 交换格式）。

hls4ml 的 ONNX 前端只接受标准算子集，因此本模块把内部 `GraphIR`
（含 rmsnorm / silu / gelu / attention / kv-cache 等自定义算子）**降级**为一组
标准 ONNX 算子的静态子图。

设计取舍（对应设计文档 hls_backend.md §2）：
- 只导出**单步 decode**（batch=1, seq=1）。自回归多步由外部逐 token 驱动，
  与 RTL 顶层 `tokk` 接口一致。
- 位置嵌入在图内作为常量切片（`pos` 是编译期已知的定值），
  避免把 RoPE / KV-cache 塞进数据通路。
- 权重以 `initializer` 形式嵌入，fp32。

导出的模型有两个额外用途：
1. 作为 hls4ml ONNX 前端的输入（仅对 hls4ml 已支持的算子子集有效）。
2. 作为**数值交叉验证**的载体：用 numpy 直接执行同一张图，与
   `rtl_backend.reference.IntModel` 的黄金向量比对。

注意 hls4ml 的 ONNX 前端有两个硬性要求（见其 `get_input_shape` /
`sanitize_layer_name`），导出时必须满足：
  1. 每个节点的每个输入都要能在 `graph.value_info` / `output` / `input`
     查到形状 —— **包括 initializer**；
  2. 每个节点都必须有非空 name。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..ir.graph import GraphIR, Node
from ..ir.ops import Op

__all__ = ["OnnxExportError", "OnnxExportResult", "export_onnx",
           "rope_matrix"]


class OnnxExportError(RuntimeError):
    """无法把 GraphIR 映射到 hls4ml 支持的 ONNX 算子集。"""


_OP = {o.value: o for o in Op}


def _op_type(node: Node) -> str:
    """取节点 op_type 的字符串部分（IR 里可能是 Op 枚举或裸字符串）。"""
    ot = node.op_type
    if isinstance(ot, Op):
        return ot.value
    ot = str(ot)
    return _OP[ot].value if ot in _OP else ot


def _safe(name: str) -> str:
    """ONNX 张量/节点名的合法化。"""
    return "".join(c if (c.isalnum() or c == "_") else "_" for c in name)


def rope_matrix(hidden: int, head_dim: int, pos: int, theta: float) -> np.ndarray:
    """构造 RoPE 的 [hidden, hidden] 旋转矩阵 R（float32）。

    约定与 `rtl_backend/reference.py::IntModel.rope` 完全一致：每个 head 是一个
    ``head_dim`` 大小的块，块内对 ``(i, i + head_dim//2)`` 这一对元素做旋转，
    角度 ``ang = pos / theta**(2*i/head_dim)``：

        out[i]            =  x[i]*cos(ang) - x[i+hd/2]*sin(ang)
        out[i + hd/2]     =  x[i]*sin(ang) + x[i+hd/2]*cos(ang)

    于是对任意向量 x 有 ``rope(x) = R @ x``。
    """
    hd = int(head_dim) or int(hidden)
    if hd <= 0 or hidden % hd != 0:
        raise OnnxExportError(
            f"head_dim={hd} 无法整除 hidden={hidden}")
    R = np.eye(hidden, dtype=np.float64)
    for start in range(0, hidden, hd):
        for i in range(hd // 2):
            a, b = i, i + hd // 2
            ang = float(pos) / (float(theta) ** (2.0 * i / hd))
            c, s = np.cos(ang), np.sin(ang)
            R[start + a, start + a] = c
            R[start + a, start + b] = -s
            R[start + b, start + a] = s
            R[start + b, start + b] = c
    return R.astype(np.float32)


@dataclass
class OnnxExportResult:
    """导出结果。"""
    path: str = ""                 # 产出的 .onnx 路径
    onnx_model: object = None      # onnx.ModelProto
    node_count: int = 0            # 映射后的 ONNX 节点数
    unsupported: list = field(default_factory=list)  # 未直接支持的内部算子
    inputs: list = field(default_factory=list)
    outputs: list = field(default_factory=list)
    embedded: list = field(default_factory=list)  # 嵌入权重的名字


def _has_onnx() -> bool:
    try:
        import onnx  # noqa: F401
        return True
    except ImportError:
        return False


def export_onnx(ir: GraphIR, out_dir: str, pos: int = 0,
                name: str = None) -> OnnxExportResult:
    """把 GraphIR 导出为单步 decode 的 ONNX 模型。

    Parameters
    ----------
    ir : GraphIR
        parser 产出的 LLM-IR（浮点权重）。
    out_dir : str
        输出目录，写入 ``<name>.onnx``。
    pos : int
        编译期已知的位置索引（decode 第 pos 步）。位置嵌入按此常量切片。
    name : str, optional
        模型名，默认取 ``ir.name``。
    """
    if not _has_onnx():
        raise OnnxExportError(
            "导出 ONNX 需要 onnx 包，请 `pip install onnx`"
            "（或 `pip install hls4ml[onnx]`）。")

    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model_name = _safe(name or ir.name or "llm2asic")
    cfg = dict(ir.config or {})

    nodes: list = []          # onnx nodes
    inits: list = []          # initializers
    used: set = set()         # 记录被嵌入的权重
    init_names: set = set()   # initializer 的 ONNX 名
    unsupported: list = []

    def wdata(wname: str) -> np.ndarray:
        w = ir.weights.get(wname)
        if w is None or w.data is None:
            raise OnnxExportError(f"权重 {wname} 缺失数值（data=None）")
        return np.asarray(w.data, dtype=np.float32)

    def add_init(wname: str, arr) -> str:
        """把数组加成 initializer，返回其 ONNX 名。"""
        safe = _safe(wname)
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=np.float32), name=safe))
        used.add(wname)
        init_names.add(safe)
        return safe

    by_op: dict = {}
    for n in ir.nodes:
        by_op.setdefault(_op_type(n), []).append(n)

    if Op.EMBEDDING.value not in by_op:
        raise OnnxExportError("GraphIR 缺少 EMBEDDING 节点，无法构造单步 decode 图")

    emb_nodes = by_op[Op.EMBEDDING.value]
    wte_node = emb_nodes[0]
    wpe_node = emb_nodes[1] if len(emb_nodes) > 1 else None

    wte_arr = wdata(wte_node.weight_names[0])
    hidden = int(wte_arr.shape[1])
    head_dim = int(cfg.get("head_dim", hidden) or hidden)
    rope_theta = float(cfg.get("rope_theta", 10000.0))

    # IR 张量名 -> ONNX 张量名
    tmap: dict = {}

    def onx(t: str) -> str:
        return tmap.get(t, _safe(t))

    # ---- 输入：单 token ----
    inp_tokens = helper.make_tensor_value_info(
        "tokens", TensorProto.INT32, [1, 1])

    # ---- 词嵌入: Gather(tokens, wte) -> reshape [1, hidden] ----
    wte = add_init(wte_node.weight_names[0], wte_arr)
    inits.append(numpy_helper.from_array(
        np.array([1, hidden], dtype=np.int64), name="emb_shape"))
    init_names.add("emb_shape")
    nodes.append(helper.make_node(
        "Gather", [wte, "tokens"], ["emb_gathered"], name="embed_gather",
        axis=0))
    nodes.append(helper.make_node(
        "Reshape", ["emb_gathered", "emb_shape"], [_safe(wte_node.outputs[0])],
        name="embed_reshape"))
    tmap[wte_node.outputs[0]] = _safe(wte_node.outputs[0])

    # ---- 位置嵌入：编译期常量切片 ----
    cur = tmap[wte_node.outputs[0]]
    if wpe_node is not None:
        wpe_arr = wdata(wpe_node.weight_names[0])
        npos = int(wpe_arr.shape[0])
        if not (0 <= pos < npos):
            raise OnnxExportError(f"pos={pos} 越界（位置嵌入只有 {npos} 行）")
        add_init("wpe_pos_slice", wpe_arr[pos:pos + 1])
        nodes.append(helper.make_node(
            "Add", [cur, "wpe_pos_slice"], [_safe(wpe_node.outputs[0])],
            name="pos_add"))
        cur = _safe(wpe_node.outputs[0])
    tmap[wpe_node.outputs[0] if wpe_node else "__nopos__"] = cur

    # ---- 主干 ----
    # 单步 decode 下需要真正展开的算子：attention 的数据依赖必须保留，
    # 否则下游 o_proj 会失去输入。rope/kv-cache 在单步图里被折叠掉。
    structural = {Op.CLONE.value, Op.COPY.value, Op.CONSTANT.value,
                  Op.RESHAPE.value, Op.PERMUTE.value, Op.TRANSPOSE.value,
                  Op.KV_STORE.value, Op.KV_LOAD.value, Op.CONCAT.value,
                  Op.UNFLATTEN.value, Op.GEMM.value, Op.GEMV.value,
                  Op.MATMUL.value, Op.BMM.value}

    n = 0

    def nn(op_type, inputs, out, **attrs):
        nonlocal n
        name = f"{_safe(op_type.lower())}_{n}"
        n += 1
        nodes.append(helper.make_node(op_type, list(inputs), [out],
                                      name=name, **attrs))
        return out

    def known(x: str) -> bool:
        """该 ONNX 张量名是否已有定义（图输入 / 已产出 / initializer）。"""
        return x == "tokens" or x in tmap.values() or x in init_names

    for node in ir.nodes:
        if node is wte_node or node is wpe_node:
            continue
        ot = _op_type(node)
        if ot in structural:
            # 结构/注意力/KV 算子在单步 decode 的浮点等价图里被跳过：
            # 它们的语义已由 parser 折叠进 q/k/v 的线性层与常量中。
            continue
        if not node.outputs:
            continue

        ins = [onx(i) for i in node.inputs if i != "pos"]
        # 只保留有定义来源的输入（丢弃 IR 里的悬空引用）
        ins = [i for i in ins if known(i)]
        o = _safe(node.outputs[0])
        tmap[node.outputs[0]] = o
        base = _safe(node.name)

        if ot == Op.LINEAR.value:
            wn = node.weight_names[0]
            W = np.ascontiguousarray(wdata(wn), dtype=np.float32)
            # IR 权重布局为 [c_out, c_in]；ONNX MatMul 需要 [c_in, c_out]
            Wn = add_init(wn, W.T)
            mm = nn("MatMul", [ins[0], Wn], o + "_mm")
            if len(node.weight_names) > 1:
                nn("Add", [mm, add_init(node.weight_names[1],
                                        np.asarray(wdata(node.weight_names[1])).reshape(-1))], o)
            else:
                nn("Identity", [mm], o)
        elif ot == Op.ADD.value:
            if len(ins) >= 2:
                nn("Add", [ins[0], ins[1]], o)
            else:
                nn("Identity", [ins[0]], o)
        elif ot == Op.MUL.value:
            nn("Mul", ins[:2], o)
        elif ot == Op.SUB.value:
            nn("Sub", ins[:2], o)
        elif ot == Op.DIV.value:
            nn("Div", ins[:2], o)
        elif ot == Op.RELU.value:
            nn("Relu", [ins[0]], o)
        elif ot in (Op.SILU.value, Op.GELU.value):
            # 无原生算子：silu 精确用 Mul(x, Sigmoid(x))；
            # gelu 用 tanh 近似展开为标准算子。
            if ot == Op.SILU.value:
                s = nn("Sigmoid", [ins[0]], o + "_sig")
                nn("Mul", [ins[0], s], o)
            else:
                # 0.5x(1+tanh(k(x+0.044715x^3))), k=sqrt(2/pi)
                p = base
                x = ins[0]
                c044 = add_init(p + "_c044715", np.float32(0.044715))
                ck = add_init(p + "_ck", np.float32(np.sqrt(2.0 / np.pi)))
                cone = add_init(p + "_one", np.float32(1.0))
                chalf = add_init(p + "_half", np.float32(0.5))
                x2 = nn("Mul", [x, x], o + "_x2")
                x3 = nn("Mul", [x2, x], o + "_x3")
                u = nn("Mul", [x3, c044], o + "_c3")
                u = nn("Add", [x, u], o + "_u1")
                u = nn("Mul", [u, ck], o + "_u2")
                t = nn("Tanh", [u], o + "_tanh")
                t = nn("Add", [t, cone], o + "_t1")
                xh = nn("Mul", [x, chalf], o + "_xh")
                nn("Mul", [xh, t], o)
        elif ot == Op.LAYERNORM.value:
            # LayerNormalization 算子本身要求 opset>=17，hls4ml 的 ONNX 前端
            # 也不支持它，因此统一降级为标准原语组合（Mean/Sub/Pow/Sqrt/Div）。
            # 这样导出的模型在 opset 13 下即可被通用运行时执行。
            p = base
            x = ins[0]
            axes = [-1]
            mu = nn("ReduceMean", [x], p + "_mu", axes=axes, keepdims=1)
            cen = nn("Sub", [x, mu], p + "_cen")
            sq = nn("Mul", [cen, cen], p + "_sq")
            var = nn("ReduceMean", [sq], p + "_var", axes=axes, keepdims=1)
            eps_n = add_init(p + "_eps",
                             np.float32(float(cfg.get("norm_eps", 1e-5))))
            v = nn("Add", [var, eps_n], p + "_ms")
            sd = nn("Sqrt", [v], p + "_sd")
            y = nn("Div", [cen, sd], p + "_norm")
            if node.weight_names:
                y = nn("Mul", [y, add_init(node.weight_names[0],
                                           np.asarray(wdata(node.weight_names[0])).reshape(-1))], p + "_g")
            if len(node.weight_names) > 1:
                y = nn("Add", [y, add_init(node.weight_names[1],
                                           np.asarray(wdata(node.weight_names[1])).reshape(-1))], o)
            else:
                nn("Identity", [y], o)
        elif ot == Op.RMSNORM.value:
            x = ins[0]
            p = base
            sq = nn("Mul", [x, x], p + "_sq")
            mean = nn("ReduceMean", [sq], p + "_mean", axes=[-1], keepdims=1)
            eps_n = add_init(p + "_eps", np.array([float(cfg.get("norm_eps", 1e-5))],
                                                  dtype=np.float32))
            m = nn("Add", [mean, eps_n], p + "_ms")
            r = nn("Sqrt", [m], p + "_rms")
            y = nn("Div", [x, r], p + "_norm")
            if node.weight_names:
                y = nn("Mul", [y, add_init(node.weight_names[0],
                                           np.asarray(wdata(node.weight_names[0])).reshape(-1))], o)
            else:
                nn("Identity", [y], o)
        elif ot == Op.SOFTMAX.value:
            nn("Softmax", [ins[0]], o, axis=-1)
        elif ot == Op.ROPE.value:
            # RoPE 是逐 head 的线性旋转（见 rtl_backend/reference.py::rope），
            # 因此可以折成常量矩阵 R：x_rope = R @ x，单个 MatMul 即可，
            # 无需 Slice/Concat 之类的复杂算子，hls4ml 也能吃下。
            R = add_init(base + "_rope_R", rope_matrix(hidden, head_dim, pos,
                                                       rope_theta))
            nn("MatMul", [ins[0], R], o)
        elif ot == Op.ATTENTION.value:
            # 单步 decode 的缩放点积注意力（KV 长度为 1，causal mask 平凡）：
            #   scores = q k^T / sqrt(head_dim) -> [1,1]
            #   out    = softmax(scores) v     -> [1, hidden]
            # head_dim 越小缩放越强；取自模型 config。
            q, k, v = ins[0], ins[1], ins[2]
            kt = nn("Transpose", [k], o + "_kt", perm=[1, 0])
            s = nn("MatMul", [q, kt], o + "_scores")
            inv = add_init(o + "_inv_sqrt_d",
                           np.array([1.0 / np.sqrt(head_dim)], dtype=np.float32))
            s = nn("Mul", [s, inv], o + "_scaled")
            a = nn("Softmax", [s], o + "_attn", axis=-1)
            nn("MatMul", [a, v], o)
        else:
            unsupported.append(f"{node.name}:{ot}")
            if ins:
                nn("Identity", [ins[0]], o)

    if not ir.outputs:
        raise OnnxExportError("GraphIR 缺少 outputs")
    final = onx(ir.outputs[0])

    out_vi = helper.make_tensor_value_info(final, TensorProto.FLOAT, [1, hidden])
    graph = helper.make_graph(nodes, f"{model_name}_graph", [inp_tokens],
                              [out_vi], initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    # ---- hls4ml 兼容处理 ----
    for init in model.graph.initializer:
        model.graph.value_info.append(helper.make_tensor_value_info(
            init.name, init.data_type, list(init.dims)))
    try:
        from onnx import shape_inference
        model = shape_inference.infer_shapes(model, strict_mode=True)
    except Exception:  # noqa: BLE001 —— 形状推断失败不致命，交给下游报错
        pass

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{model_name}.onnx")
    onnx.checker.check_model(model)
    onnx.save(model, path)

    return OnnxExportResult(path=path, onnx_model=model, node_count=len(nodes),
                            unsupported=unsupported, inputs=["tokens"],
                            outputs=[final], embedded=sorted(used))
