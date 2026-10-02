# llm2asic/hls_backend/hls4ml_gen.py
"""GraphIR -> hls4ml 模型 -> HLS C++ 工程。

hls4ml 的 **ONNX 前端不支持** `Gather` / `LayerNormalization` / `Sqrt` /
`ReduceMean`（见其 `get_supported_onnx_layers()`），而这些恰好是 LLM 的核心算子。
因此本模块绕过 ONNX 前端，直接用 hls4ml 的**原生 layer-list API**
（`hls4ml.model.ModelGraph.from_layer_list`）构造模型，从而用上 hls4ml 的
`Embedding` / `LayerNormalization` / `Dense` / `Activation` 等完整层集合。

产���（`compile()` 的生成阶段，即 hls4ml project）：
    <out>/hls4ml/<model>.cpp        顶层 HLS C++ 源（可直接喂给 Bambu）
    <out>/hls4ml/<model>_test.cpp   C 测试台
    <out>/hls4ml/firmware/...       ap_types / 权重 / parameters.h
    <out>/hls4ml/*.tcl              Vivado HLS 工程脚本

注意：`ModelGraph.compile()` 在生成之后会**调用目标 HLS 工具**（Vivado HLS），
本环境无 Vivado，因此这里用 `SkipOptimizers` + 直接调用 backend 的生成阶段，
只做"生成 C++"，把"跑 HLS 工具"这一步交给下游（Bambu，或用户的 Vivado）。

对应设计文档 hls_backend.md §3。
"""

from __future__ import annotations

import contextlib
import os
import re
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

from ..ir.graph import GraphIR, Node
from ..ir.ops import Op

__all__ = ["Hls4mlError", "Hls4mlResult", "Hls4mlConfig", "build_hls4ml"]


class Hls4mlError(RuntimeError):
    """hls4ml 未安装，或 GraphIR 无法映射到 hls4ml 层集合。"""


_OP = {o.value: o for o in Op}


def _op_type(node: Node) -> str:
    ot = node.op_type
    if isinstance(ot, Op):
        return ot.value
    ot = str(ot)
    return _OP[ot].value if ot in _OP else ot


def _safe(name: str) -> str:
    return "".join(c if (c.isalnum() or c == "_") else "_" for c in name)


@dataclass
class Hls4mlConfig:
    """hls4ml 工程配置。"""
    project_name: str = ""           # 空 = 取模型名（否则多模型会共用同一工程目录）
    out_dir: str = "hls4ml"
    target: str = "vivado"
    part: str = "xczu7ev-ffvc1156-2-e"
    clock_ns: float = 5.0
    uncertainty: float = 1.0
    precision: str = "float"       # float | fixed / ap_fixed<W,I>
    reuse_factor: int = 1
    strategy: str = "Latency"
    io_type: str = "io_parallel"


@dataclass
class Hls4mlResult:
    """hls4ml 生成结果。"""
    project_dir: str = ""
    cpp_path: str = ""            # 顶层 HLS C++（可喂给 Bambu）
    test_cpp_path: str = ""
    firmware_dir: str = ""
    layer_count: int = 0
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and bool(self.cpp_path)


def _has_hls4ml() -> bool:
    try:
        import hls4ml  # noqa: F401
        from hls4ml.model import ModelGraph  # noqa: F401
        return True
    except ImportError:
        return False


def _ep_power(eps: float) -> int:
    """把 epsilon 转成 hls4ml LayerNormalization 的 epsilon_power_of_10。"""
    # 1e-5 -> 5
    e = int(round(np.log10(max(eps, 1e-12))))
    return int(e)


@contextlib.contextmanager
def _capture_output(max_bytes: int = 1 << 20):
    """在 fd 级别捕获 stdout/stderr。

    hls4ml 的 compile() 会 fork 出 g++ 子进程，那些输出不经过
    contextlib.redirect_stderr，所以必须在 fd 层面重定向。
    超出 max_bytes 后停止累积，避免 hls4ml 报错时吞掉大量内存。
    """
    import sys
    import tempfile

    res: dict = {"text": "", "truncated": False}
    saved_out, saved_err = os.dup(1), os.dup(2)
    fd, path = tempfile.mkstemp(prefix="hls4ml_compile_", suffix=".log")
    try:
        with os.fdopen(fd, "wb") as sink:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            try:
                yield res
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
                os.dup2(saved_out, 1)
                os.dup2(saved_err, 2)
        size = os.path.getsize(path)
        with open(path, "r", errors="replace") as fh:
            res["text"] = fh.read(max_bytes)
        res["truncated"] = size > max_bytes
    finally:
        os.close(saved_out)
        os.close(saved_err)
        try:
            os.unlink(path)
        except OSError:
            pass


def build_layers(ir: GraphIR, cfg: Hls4mlConfig,
                 pos: int = 0) -> tuple:
    """把 GraphIR 映射成 hls4ml layer list。

    返回 ``(layer_list, meta)``；``meta['unsupported']`` 列出被跳过的算子。
    """
    mcfg = dict(ir.config or {})
    hidden = None
    layers: list = []
    unsupported: list = []
    tmap: dict = {}     # IR 张量名 -> hls4ml 层名

    def wdata(wn: str) -> np.ndarray:
        w = ir.weights.get(wn)
        if w is None or w.data is None:
            raise Hls4mlError(f"权重 {wn} 缺失数值（data=None）")
        return np.asarray(w.data, dtype=np.float32)

    by_op: dict = {}
    for n in ir.nodes:
        by_op.setdefault(_op_type(n), []).append(n)

    emb_nodes = by_op.get(Op.EMBEDDING.value, [])
    if not emb_nodes:
        raise Hls4mlError("GraphIR 缺少 EMBEDDING 节点")

    wte_node = emb_nodes[0]
    wte_arr = wdata(wte_node.weight_names[0])
    hidden = int(wte_arr.shape[1])
    vocab = int(wte_arr.shape[0])

    # hls4ml 要求每层张量形状自洽（[1, N]），因此必须跟踪每个中间张量的宽度：
    # LLM 内部宽度会变化（MLP 是 hidden -> n_inner -> hidden）。
    wmap: dict = {}

    def w_of(layer_name: str) -> int:
        return int(wmap.get(layer_name, hidden))

    # --- 输入 + 词嵌入 ---
    layers.append({'name': 'tokens', 'class_name': 'InputLayer',
                   'input_shape': [1]})   # hls4ml 会自动补 batch 维 -> [1,1]
    layers.append({'name': 'embed', 'class_name': 'Embedding',
                   'n_in': 1, 'n_out': hidden, 'vocab_size': vocab,
                   'embeddings_data': wte_arr})
    wmap['tokens'] = 1
    wmap['embed'] = hidden
    tmap[wte_node.outputs[0]] = 'embed'

    # --- 位置嵌入：编译期常量，直接做成 Constant 层求和 ---
    cur = 'embed'
    if len(emb_nodes) > 1:
        wpe_node = emb_nodes[1]
        wpe_arr = wdata(wpe_node.weight_names[0])
        if not (0 <= pos < wpe_arr.shape[0]):
            raise Hls4mlError(f"pos={pos} 越界（位置嵌入 {wpe_arr.shape[0]} 行）")
        layers.append({'name': 'posvec', 'class_name': 'Constant',
                       'value': np.ascontiguousarray(
                           wpe_arr[pos:pos + 1], dtype=np.float32)})
        layers.append({'name': 'posadd', 'class_name': 'Merge',
                       'op': 'add', 'inputs': [cur, 'posvec']})
        wmap['posvec'] = hidden
        wmap['posadd'] = hidden
        cur = 'posadd'
        tmap[wpe_node.outputs[0]] = cur

    # --- 主干 ---
    for n in ir.nodes:
        if n is wte_node or (len(emb_nodes) > 1 and n is emb_nodes[1]):
            continue
        ot = _op_type(n)
        if ot in (Op.CLONE.value, Op.COPY.value, Op.CONSTANT.value,
                  Op.RESHAPE.value, Op.PERMUTE.value, Op.TRANSPOSE.value,
                  Op.KV_STORE.value, Op.KV_LOAD.value, Op.CONCAT.value,
                  Op.UNFLATTEN.value, Op.GEMM.value, Op.GEMV.value,
                  Op.MATMUL.value, Op.BMM.value, Op.ROPE.value):
            continue

        ins = [tmap[i] for i in n.inputs if i in tmap and i != "pos"]
        if not ins or not n.outputs:
            continue
        out_layer = None
        base = _safe(n.name)

        if ot == Op.LINEAR.value:
            wn = n.weight_names[0]
            # IR 权重布局为 [c_out, c_in]（见 rtl_backend/numeric.py::QWeight）
            W = np.ascontiguousarray(wdata(wn), dtype=np.float32)
            n_out, n_in_w = int(W.shape[0]), int(W.shape[1])
            n_in = w_of(ins[0])
            if n_in_w != n_in:
                raise Hls4mlError(
                    f"{n.name}: 权重输入维 {n_in_w} 与张量宽度 {n_in} 不符")
            layers.append({'name': base, 'class_name': 'Dense',
                           'n_in': n_in, 'n_out': n_out,
                           'weight_data': W,
                           'inputs': [ins[0]]})
            wmap[base] = n_out
            if len(n.weight_names) > 1:
                b = np.asarray(wdata(n.weight_names[1]),
                               dtype=np.float32).reshape(1, -1)
                layers.append({'name': base + '_b', 'class_name': 'Constant',
                               'value': b})
                wmap[base + '_b'] = n_out
                layers.append({'name': base + '_ba', 'class_name': 'Merge',
                               'op': 'add', 'inputs': [base, base + '_b']})
                wmap[base + '_ba'] = n_out
                out_layer = base + '_ba'
            else:
                out_layer = base
        elif ot == Op.ADD.value:
            if len(ins) >= 2:
                layers.append({'name': base, 'class_name': 'Merge',
                               'op': 'add', 'inputs': ins[:2]})
                wmap[base] = max(w_of(i) for i in ins[:2])
                out_layer = base
            else:
                layers.append({'name': base, 'class_name': 'Activation',
                               'activation': 'linear', 'inputs': [ins[0]]})
                wmap[base] = w_of(ins[0])
                out_layer = base
        elif ot == Op.MUL.value:
            layers.append({'name': base, 'class_name': 'Merge',
                           'op': 'multiply', 'inputs': ins[:2]})
            wmap[base] = max(w_of(i) for i in ins[:2])
            out_layer = base
        elif ot in (Op.SILU.value, Op.GELU.value):
            # hls4ml 无 silu/gelu 激活；用 Sigmoid + Mul 展开
            #   silu(x) = x * sigmoid(x)         （精确）
            #   gelu(x) ≈ 0.5x(1+tanh(...))      （tanh 近似，逐项展开）
            x = ins[0]
            nw = w_of(x)
            if ot == Op.SILU.value:
                layers.append({'name': base + '_sig', 'class_name': 'Activation',
                               'activation': 'sigmoid', 'inputs': [x]})
                layers.append({'name': base, 'class_name': 'Merge',
                               'op': 'multiply', 'inputs': [x, base + '_sig']})
            else:
                # 0.5*x*(1+tanh(k*(x+0.044715x^3)))
                k = float(np.sqrt(2.0 / np.pi))
                for nm, c in (('c044', 0.044715), ('ck', k),
                              ('chalf', 0.5), ('cone', 1.0)):
                    layers.append({'name': f'{base}_{nm}',
                                   'class_name': 'Constant',
                                   'value': np.full((1, nw), c, dtype=np.float32)})
                    wmap[f'{base}_{nm}'] = nw

                def merge(name, op, inputs):
                    layers.append({'name': name, 'class_name': 'Merge',
                                   'op': op, 'inputs': list(inputs)})
                    wmap[name] = nw
                    return name

                x2 = merge(base + '_x2', 'multiply', [x, x])
                x3 = merge(base + '_x3', 'multiply', [x2, x])
                c3 = merge(base + '_c3', 'multiply', [x3, f'{base}_c044'])
                u1 = merge(base + '_u1', 'add', [x, c3])
                u2 = merge(base + '_u2', 'multiply', [u1, f'{base}_ck'])
                layers.append({'name': base + '_t', 'class_name': 'Activation',
                               'activation': 'tanh', 'inputs': [u2]})
                wmap[base + '_t'] = nw
                p1 = merge(base + '_p1', 'add', [base + '_t', f'{base}_cone'])
                xh = merge(base + '_xh', 'multiply', [x, f'{base}_chalf'])
                merge(base, 'multiply', [xh, p1])
            out_layer = base
        elif ot == Op.LAYERNORM.value:
            n_in = w_of(ins[0])
            g = np.asarray(wdata(n.weight_names[0]),
                           dtype=np.float32).reshape(-1)
            b = (np.asarray(wdata(n.weight_names[1]), dtype=np.float32).reshape(-1)
                 if len(n.weight_names) > 1 else np.zeros_like(g))
            if g.shape[0] != n_in:
                raise Hls4mlError(
                    f"{n.name}: gamma 长度 {g.shape[0]} 与宽度 {n_in} 不符")
            layers.append({'name': base, 'class_name': 'LayerNormalization',
                           'n_in': n_in, 'seq_len': 1,
                           'gamma_data': g, 'beta_data': b,
                           'epsilon_power_of_10': _ep_power(
                               float(mcfg.get("norm_eps", 1e-5))),
                           'inputs': [ins[0]]})
            wmap[base] = n_in
            out_layer = base
        elif ot == Op.RMSNORM.value:
            # hls4ml 的 LayerNormalization 是"减均值"语义，不是 RMSNorm。
            # 退化为等价的 scale-only 形式：hls4ml 无原生 RMSNorm，
            # 这里用 Mul(x, gamma) 保留仿射部分，归一化部分由下游定点核承担。
            n_in = w_of(ins[0])
            g = (np.asarray(wdata(n.weight_names[0]), dtype=np.float32)
                 .reshape(1, -1) if n.weight_names
                 else np.ones((1, n_in), dtype=np.float32))
            layers.append({'name': base + '_g', 'class_name': 'Constant',
                           'value': g})
            wmap[base + '_g'] = n_in
            layers.append({'name': base, 'class_name': 'Merge',
                           'op': 'multiply', 'inputs': [ins[0], base + '_g']})
            wmap[base] = n_in
            unsupported.append(f"{n.name}: rmsnorm(hls4ml 无原生 RMSNorm，退化为 gamma 缩放)")
            out_layer = base
        elif ot == Op.SOFTMAX.value:
            layers.append({'name': base, 'class_name': 'Activation',
                           'activation': 'softmax', 'inputs': [ins[0]]})
            wmap[base] = w_of(ins[0])
            out_layer = base
        elif ot == Op.ATTENTION.value:
            # hls4ml 无注意力层；单步 decode 下 KV 长度=1，注意力退化为
            # softmax(单标量)·v = v，故直接透传 v。
            # 注意：RoPE 已被折叠掉，此时 ins 可能少于 3 个，v 取最后一个已知输入。
            if len(ins) < 3:
                unsupported.append(
                    f"{n.name}: attention 输入不完整({ins})，退化为透传")
            v_src = ins[2] if len(ins) >= 3 else ins[-1]
            layers.append({'name': base, 'class_name': 'Activation',
                           'activation': 'linear', 'inputs': [v_src]})
            wmap[base] = w_of(v_src)
            unsupported.append(f"{n.name}: attention(单步 decode 退化为 v 透传)")
            out_layer = base
        elif ot in (Op.RELU.value,):
            layers.append({'name': base, 'class_name': 'Activation',
                           'activation': 'relu', 'inputs': [ins[0]]})
            wmap[base] = w_of(ins[0])
            out_layer = base
        elif ot == Op.SUB.value:
            layers.append({'name': base, 'class_name': 'Merge',
                           'op': 'subtract', 'inputs': ins[:2]})
            wmap[base] = max(w_of(i) for i in ins[:2])
            out_layer = base
        elif ot == Op.DIV.value:
            layers.append({'name': base, 'class_name': 'Merge',
                           'op': 'divide', 'inputs': ins[:2]})
            wmap[base] = max(w_of(i) for i in ins[:2])
            out_layer = base
        else:
            unsupported.append(f"{n.name}:{ot}")
            layers.append({'name': base, 'class_name': 'Activation',
                           'activation': 'linear', 'inputs': [ins[0]]})
            wmap[base] = w_of(ins[0])
            out_layer = base

        if out_layer and n.outputs:
            tmap[n.outputs[0]] = out_layer

    return layers, {'unsupported': unsupported, 'hidden': hidden,
                    'vocab': vocab, 'final': tmap.get(ir.outputs[0])}


def build_hls4ml(ir: GraphIR, out_dir: str, cfg: Optional[Hls4mlConfig] = None,
                 pos: int = 0) -> Hls4mlResult:
    """生成 hls4ml HLS C++ 工程。

    Parameters
    ----------
    ir : GraphIR
        parser 产出的 LLM-IR。
    out_dir : str
        输出根目录；工程写到 ``<out_dir>/hls4ml/``。
    cfg : Hls4mlConfig, optional
    pos : int
        编译期位置索引（decode 第 pos 步）。
    """
    res = Hls4mlResult()
    if not _has_hls4ml():
        res.errors.append(
            "未安装 hls4ml。请 `pip install \"hls4ml[onnx]\"`。")
        return res

    # 用副本：project_name 是派生值，不能写回调用方复用的 config
    hcfg = replace(cfg or Hls4mlConfig(),
                   project_name=_safe((cfg.project_name if cfg else "")
                                      or ir.name or "llm2asic"))
    # 每个模型独占一个子目录：hls4ml 会把 firmware/权重/*.tcl 平铺到 OutputDir，
    # 多模型共用同一目录会互相覆盖。
    proj_dir = os.path.join(out_dir, hcfg.out_dir, hcfg.project_name)
    os.makedirs(proj_dir, exist_ok=True)

    try:
        layers, meta = build_layers(ir, hcfg, pos=pos)
    except Hls4mlError as e:
        res.errors.append(str(e))
        return res
    res.warnings.extend(meta.get("unsupported", []))

    final = meta.get("final")
    if not final:
        res.errors.append("未能确定 hls4ml 图的输出层")
        return res

    from hls4ml.model import ModelGraph

    pcfg = {
        'OnnxModel': None,
        'OutputDir': proj_dir,
        'IOType': hcfg.io_type,
        'ProjectName': hcfg.project_name,
        'ProjectDir': os.path.join(proj_dir, f"{hcfg.project_name}_prj"),
        'Target': hcfg.target,
        'Part': hcfg.part,
        'Clock': {'Period': hcfg.clock_ns, 'Uncertainty': hcfg.uncertainty},
        'Model': {'Precision': hcfg.precision, 'ReuseFactor': hcfg.reuse_factor,
                  'Strategy': hcfg.strategy},
        'HLSConfig': {'Model': {'Precision': hcfg.precision,
                                'ReuseFactor': hcfg.reuse_factor,
                                'Strategy': hcfg.strategy}},
    }

    try:
        hls_model = ModelGraph.from_layer_list(pcfg, layers)
        # compile() 会在生成之后调用目标 HLS 工具（Vivado HLS），
        # 本环境无 Vivado；用 SkipOptimizers 关闭优化后仍会尝试跑工具，
        # 因此这里捕获该异常 —— 生成阶段（写 C++）已经完成。
        #
        # hls4ml 内部会直接调用 g++ 编译 firmware/（其 LayerNorm 在浮点
        # 模式下有歧义重载，见下方 _fix_float_layernorm_table），
        # 错误信息会淹没输出，这里把 fd 级 stdout/stderr 收进临时文件。
        with _capture_output() as cap:
            try:
                hls_model.compile()
            except Exception as e:  # noqa: BLE001
                res.warnings.append(
                    f"目标 HLS 工具({hcfg.target})不可用，已仅生成 C++ 工程: "
                    f"{type(e).__name__}: {e}")
        if cap.get("text", "").strip():
            res.warnings.append(
                f"hls4ml compile 阶段输出（末 3 行）: "
                f"{' | '.join(cap['text'].strip().splitlines()[-3:])[:300]}")
    except Exception as e:  # noqa: BLE001
        res.errors.append(f"hls4ml 生成失败: {type(e).__name__}: {e}")
        return res

    res.project_dir = proj_dir
    res.layer_count = len(layers)
    cpp = os.path.join(proj_dir, f"{hcfg.project_name}_bridge.cpp")
    if os.path.exists(cpp):
        res.cpp_path = cpp
    test = os.path.join(proj_dir, f"{hcfg.project_name}_test.cpp")
    if os.path.exists(test):
        res.test_cpp_path = test
    fw = os.path.join(proj_dir, "firmware")
    if os.path.isdir(fw):
        res.firmware_dir = fw

    if not res.cpp_path:
        res.errors.append(
            f"hls4ml 未产出顶层 C++（期望 {cpp}）")
        return res

    # hls4ml 1.3.0 的浮点模式缺陷：LayerNormalization 的查表类型 table_t
    # 仍是 ap_ufixed<8,5>，而 accum_t 已是 float，导致
    #   data_diff[i] * deno_inver * scale[i]
    # 触发 `ambiguous overload for operator*`，生成的 C++ 根本无法用 g++ 编译。
    # 这里把 layernorm config 的 table_t 统一改成模型默认类型（float）。
    if hcfg.precision in ("float", "double"):
        n = _fix_float_layernorm_table(proj_dir)
        if n:
            res.warnings.append(
                f"已修正 {n} 处 LayerNorm config 的 table_t 为 "
                f"model_default_t（hls4ml 浮点模式下的 ap_ufixed 歧义重载问题）")
    return res


_LAYERNORM_STRUCT = re.compile(
    r"(struct\s+config\d+\s*:\s*nnet::layernorm_config\s*\{)(.*?)(\n\};)",
    re.S)
_TABLE_T_LINE = re.compile(r"^(\s*typedef\s+)[A-Za-z0-9_]+_table_t(\s+table_t;)\s*$",
                           re.M)


def _fix_float_layernorm_table(project_dir: str) -> int:
    """把生成的 parameters.h 里 layernorm config 的 `table_t` 改为 model_default_t。

    返回修改处数。hls4ml 版本升级后若不再需要，应连同调用点一起移除。
    """
    path = os.path.join(project_dir, "firmware", "parameters.h")
    if not os.path.isfile(path):
        return 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return 0

    count = 0

    def repl(m):
        nonlocal count
        head, body, tail = m.group(1), m.group(2), m.group(3)
        new_body, k = _TABLE_T_LINE.subn(
            r"\1model_default_t\2", body)
        count += k
        return head + new_body + tail

    new_text = _LAYERNORM_STRUCT.sub(repl, text)
    if count and new_text != text:
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_text)
        except OSError:
            return 0
    return count
