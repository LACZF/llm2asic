# llm2asic/hls_backend/c_kernel.py
"""GraphIR -> Bambu 可综合的 HLS C/C++ 内核（单步 decode）。

Bambu（PandA HLS）以 **C/C++ 源码**为输入，做循环变换 / 流水化 / 存储划分后
生成 Verilog。本模块把内部 `GraphIR` 翻译成一个自包含的 C++ 内核：

- 顶层函数 ``<prefix>_top(int token, int pos, T *out)``：单步 decode；
- 权重以 C 数组字面量内联（const 静态表，Bambu 会识别为 ROM/BRAM）；
- 带 Bambu 识别的 HLS pragma（只用 PandA 插件真正支持的子集：函数作用域
  ``PIPELINE`` 与 ``INTERFACE mode=... port=...``；见 ``_PIPE`` 处的说明）；
- ``#ifndef SYNTHESIS`` 下附带 ``main()``，用于本机 g++ 校验与 Bambu ``--simulate``。

数值语义与 `onnx_export.py` 一致（同一张单步 decode 图）。

实现要点：中间张量通过**线性扫描寄存器分配**映射到少量 C 缓冲区，
只有在最后一个消费者之后才复用，避免直线化展开时覆盖仍存活的中间值。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

from ..ir.graph import GraphIR, Node
from ..ir.ops import Op

__all__ = ["CKernelError", "CKernelConfig", "CKernelResult", "gen_c_kernel"]


class CKernelError(RuntimeError):
    """无法把 GraphIR 翻译成 C 内核。"""


def _op_type(node: Node) -> str:
    ot = node.op_type
    if isinstance(ot, Op):
        return ot.value
    s = str(ot)
    return s


def _safe(name: str) -> str:
    out = []
    for i, c in enumerate(str(name)):
        if c.isalnum() or c == "_":
            out.append(c)
        else:
            out.append("_")
    s = "".join(out)
    if s and s[0].isdigit():
        s = "_" + s
    return s or "t"


# 直接透传 / 由顶层函数内联处理的算子，不需要单独发射
# 注意：ROPE 不在此列 —— 它有专门的发射分支（常量旋转矩阵）。
_SKIPPED = {
    Op.CLONE.value, Op.COPY.value, Op.CONSTANT.value, Op.RESHAPE.value,
    Op.PERMUTE.value, Op.TRANSPOSE.value, Op.KV_STORE.value, Op.KV_LOAD.value,
    Op.CONCAT.value, Op.UNFLATTEN.value, Op.GEMM.value, Op.GEMV.value,
    Op.MATMUL.value, Op.BMM.value,
}


@dataclass
class CKernelConfig:
    """C 内核生成配置。"""
    prefix: str = ""                # 顶层/文件名前缀，空 = 取模型名
    precision: str = "float"        # float | double
    pipeline: int = 0                 # >0 -> `PIPELINE II=N`；0(默认) -> 不发 PIPELINE。
                                     # 这些 kernel 的访存延迟约 30ns，任何形式的
                                     # 函数级 PIPELINE 都会调度失败
                                     # ("Timing of Vertex ... is not compatible
                                     # with II=1. Actual vertex latency is 34.1
                                     # greater than the clock period")，
                                     # 5ns 和 40ns 下都一样。先让 Bambu 自由
                                     # 调度出正确 RTL，流水化留给后续调优。
    array_partition: bool = True
    emit_main: bool = True
    n_buffers: int = 8              # 每种宽度类别的缓冲池上限；打满且无空闲
                                   # 缓冲时报错（绝不复用活跃缓冲）
    trace: bool = False             # 生成逐算子 TRACE printf（调试用）


@dataclass
class CKernelResult:
    """生成结果。"""
    top_path: str = ""
    top_fname: str = ""
    vocab: int = 0
    hidden: int = 0
    buffers: int = 0
    line_count: int = 0
    trace: list = field(default_factory=list)   # (算子, 输入C名, 输出C名, 宽度)
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and bool(self.top_path)


def _c(v, sf: str = "f") -> str:
    """浮点字面量。sf 为类型后缀（float -> "f"，double -> 空）。

    注意：double 模式下绝不能加 "f"，否则 eps 等常量会被截断成单精度
    （float(1e-5) != 1e-5），逐位比对时会出现 1e-9 量级的系统偏差。
    """
    f = float(v)
    if f != f:
        return "0.0" + sf
    if f in (float("inf"), float("-inf")):
        return ("3.402823466e+38" if f > 0 else "-3.402823466e+38") + sf
    t = repr(f)
    if "e" in t or "E" in t:
        return t + sf
    if "." not in t:
        t += ".0"
    return t + sf


def _fmt(vals, per_line: int = 6, sf: str = "f") -> str:
    rows = []
    for i in range(0, len(vals), per_line):
        rows.append("    " + ", ".join(_c(v, sf) for v in vals[i:i + per_line]))
    return ",\n".join(rows)


class _Alloc:
    """按宽度类别分配的 C 缓冲区池，带活跃区间复用。"""

    def __init__(self, max_buffers: int = 8):
        self.max_buffers = max_buffers
        self.pool: dict = {}        # class_key -> list[(cname, width, last_use)]
        self.count = 0

    def alloc(self, width: int, idx: int, last_use: int) -> str:
        key = width
        free = [b for b in self.pool.get(key, []) if b[2] < idx]
        if free:
            free.sort(key=lambda b: b[2])
            b = free[0]
            b[2] = last_use
            return b[0]
        lst = self.pool.setdefault(key, [])
        if len(lst) >= self.max_buffers:
            # 没有已失活的同宽缓冲可复用，而池已达上限。
            # 此时“复用”只能挑一个仍活跃的缓冲，会在生成代码里静默覆盖
            # 活跃数据；宁可明确报错，也不要产出数值错误的内核。
            raise CKernelError(
                f"宽度 {width} 的缓冲池已达上限 n_buffers={self.max_buffers}，"
                f"且没有可复用的空闲缓冲（节点 #{idx}）。"
                f"请调大 CKernelConfig.n_buffers。")
        self.count += 1
        cname = f"b{self.count}"
        lst.append([cname, width, last_use])
        return cname

    def decls(self) -> list:
        out = []
        for key, lst in sorted(self.pool.items()):
            for cname, width, _ in sorted(lst, key=lambda x: x[0]):
                out.append((cname, width))
        return sorted(out)


def gen_c_kernel(ir: GraphIR, out_dir: str,
                 cfg: Optional[CKernelConfig] = None) -> CKernelResult:
    """生成 Bambu 输入用的 HLS C++ 内核。"""
    # 用副本，避免把 prefix 等派生值写回调用方复用的 config
    kcfg = replace(cfg or CKernelConfig(),
                   prefix=_safe((cfg.prefix if cfg else "") or ir.name
                                or "llm2asic"))
    res = CKernelResult(top_fname=f"{kcfg.prefix}_top")

    mcfg = dict(ir.config or {})
    T = "double" if kcfg.precision == "double" else "float"
    SF = "" if T == "double" else "f"
    LSF = SF

    def wdata(wn: str) -> np.ndarray:
        w = ir.weights.get(wn)
        if w is None or w.data is None:
            raise CKernelError(f"权重 {wn} 缺失数值（data=None）")
        return np.asarray(w.data)

    # ---------- 权重与配置 ----------
    by_op: dict = {}
    for n in ir.nodes:
        by_op.setdefault(_op_type(n), []).append(n)
    if Op.EMBEDDING.value not in by_op:
        raise CKernelError("GraphIR 缺少 EMBEDDING 节点")
    emb = by_op[Op.EMBEDDING.value]
    wte = np.asarray(wdata(emb[0].weight_names[0]), dtype=np.float64)
    vocab, hidden = int(wte.shape[0]), int(wte.shape[1])
    head_dim = int(mcfg.get("head_dim", hidden) or hidden)
    eps = float(mcfg.get("norm_eps", 1e-5) or 1e-5)
    rope_theta = float(mcfg.get("rope_theta", 10000.0))
    res.vocab, res.hidden = vocab, hidden

    # GPT-2 风格有第二个 EMBEDDING（位置嵌入）；Llama 用 RoPE 而没有 wpe。
    emb_prologue = emb[:2] if len(emb) >= 2 else emb[:1]
    emb_ins = {id(e) for e in emb_prologue}
    wpe = (np.asarray(wdata(emb_prologue[1].weight_names[0]), dtype=np.float64)
           if len(emb_prologue) >= 2 else None)

    # ---------- 待发射节点 ----------
    # 跳过内联/透传算子。"" 是缺省 op_type（无类型节点）也要跳过。
    skip = _SKIPPED | {""}
    active = [n for n in ir.nodes
              if id(n) not in emb_ins and _op_type(n) not in skip
              and getattr(n, "outputs", None)]

    # 输出张量宽度
    def out_width(n: Node) -> int:
        ot = _op_type(n)
        w0 = width_of.get(n.inputs[0], hidden) if n.inputs else hidden
        if ot == Op.LINEAR.value:
            return int(np.asarray(wdata(n.weight_names[0])).shape[0])
        if ot in (Op.ADD.value, Op.MUL.value, Op.SUB.value):
            ws = [width_of.get(i, hidden) for i in n.inputs[:2]]
            return max(ws) if ws else w0
        if ot == Op.ATTENTION.value:
            for i in n.inputs:
                if i in width_of:
                    wi = width_of[i]
            vsrc = None
            for i in n.inputs[2:]:
                if i in width_of:
                    vsrc = i
                    break
            if vsrc is None:
                for i in n.inputs:
                    if i in width_of:
                        vsrc = i
            return width_of.get(vsrc, hidden) if vsrc else w0
        return w0

    width_of: dict = {}
    for n in active:
        for i in n.inputs:
            if i not in width_of and i in {m.outputs[0] for m in ir.nodes if m.outputs}:
                width_of.setdefault(i, hidden)
    for e in emb_prologue:
        width_of[e.outputs[0]] = hidden
    for n in active:
        width_of[n.outputs[0]] = out_width(n)

    # 最后使用位置（用于缓冲区复用）
    last_use: dict = {}
    for idx, n in enumerate(active):
        for i in n.inputs:
            last_use[i] = idx

    # ---------- 分配缓冲并建立张量 -> C 名字 映射 ----------
    # 词嵌入/位置嵌入各自独占缓冲：GPT-2 的 p0 = add(h_tok, h_pos) 必须读到
    # 两个不同数组，若融合成同一缓冲会退化成 2*h。
    alloc = _Alloc(kcfg.n_buffers)
    cmap: dict = {}
    for e in emb_prologue:
        cmap[e.outputs[0]] = alloc.alloc(hidden, -1, len(active) + 1)
    for idx, n in enumerate(active):
        o = n.outputs[0]
        cmap[o] = alloc.alloc(width_of[o], idx, last_use.get(o, idx))
    res.buffers = alloc.count

    # ---------- 常量表 ----------
    consts: dict = {}

    def emit_const(key: str, arr) -> str:
        name = f"W_{_safe(key)}"
        if name in consts:
            return name
        flat = np.asarray(arr, dtype=T).reshape(-1)
        consts[name] = flat
        return name

    wte_n = emit_const("wte", wte)
    wpe_n = emit_const("wpe", wpe) if wpe is not None else None

    # 预注册所有节点权重，之后再统一排版常量表
    wconsts: dict = {}
    # 无 gamma 的 RMSNorm 需要全 1 表
    ones_n = "W_ones_h%d" % hidden
    need_ones = any(_op_type(n) == Op.RMSNORM.value and not n.weight_names
                    for n in active)
    for n in active:
        if not n.weight_names:
            continue
        for wi, wn in enumerate(n.weight_names):
            if f"W_{_safe(wn)}" in consts:
                continue
            arr = np.asarray(wdata(wn))
            if _op_type(n) == Op.LINEAR.value and wi == 0:
                # 不转置：gemv 用 w[o*n_in + i] 访问，恰好就是 IR 的
                # [c_out, c_in] 行主序布局，转置反而会算错。
                arr = np.ascontiguousarray(arr)
            wconsts[wn] = emit_const(wn, arr)

    if need_ones:
        emit_const("ones_h%d" % hidden, np.ones(hidden))

    const_lines: list = []
    for name, flat in consts.items():
        const_lines.append(f"static const {T} {name}[{flat.size}] = {{")
        const_lines.append(_fmt(list(flat), sf=LSF))
        const_lines.append("};")

    os.makedirs(out_dir, exist_ok=True)
    L: list = []
    a = L.append

    a("/* Auto-generated by llm2asic (hls_backend/c_kernel.py).")
    a(" * Single decode step of an LLM as a Bambu/PandA HLS kernel.")
    a(" * Do not edit by hand. */")
    a("#include <math.h>")
    a("#include <stdio.h>")
    a("#include <stdlib.h>")
    a("")
    a(f"#define VOCAB {vocab}")
    a(f"#define HIDDEN {hidden}")
    a(f"#define HEAD_DIM {head_dim}")
    a("")
    a("/* ---------------- weight tables ---------------- */")
    L.extend(const_lines)
    a("")
    a("/* ---------------- inline kernels ---------------- */")
    # PandA 的 clang 插件(plugin_ASTAnalyzer)只认这几个 pragma:
    #   pipeline / inline / unroll / dataflow / cache / interface
    # 其中 PIPELINE **只允许函数作用域**（写进 for 循环体会报
    # "Loop pipelining pragma not supported."）；ARRAY_PARTITION 根本没有
    # handler（退化成 "Unknown HLS pragma" 警告）；INTERFACE 必须是
    # `mode=<mode> port=<name>` 键值形式（写成裸 `ap_none` 会报
    # "Missing interface mode attribute"）。所以统一用最小可用子集。
    PIPE = f"    #pragma HLS PIPELINE II={kcfg.pipeline}" if kcfg.pipeline > 0 \
        else None

    def pipe():
        if PIPE:
            a(PIPE)

    a(f"static void gemv(const {T} *w, int n_out, int n_in, const {T} *x, {T} *y) {{")
    pipe()
    a("    for (int o = 0; o < n_out; o++) {")
    a(f"        {T} acc = 0.0{SF};")
    a("        for (int i = 0; i < n_in; i++) {")
    a("            acc += w[o * n_in + i] * x[i];")
    a("        }")
    a("        y[o] = acc;")
    a("    }")
    a("}")
    a("")
    a(f"static void rmsnorm(const {T} *x, int n, const {T} *g, {T} eps, {T} *y) {{")
    pipe()
    a(f"    {T} ss = 0.0{SF};")
    a("    for (int i = 0; i < n; i++) {")
    a("        ss += x[i] * x[i];")
    a("    }")
    a(f"    {T} inv = 1.0{SF} / sqrt(ss / ({T})n + eps);")
    a("    for (int i = 0; i < n; i++) {")
    a("        y[i] = x[i] * inv * g[i];")
    a("    }")
    a("}")
    a("")
    a(f"static void layernorm(const {T} *x, int n, const {T} *g, const {T} *b, {T} eps, {T} *y) {{")
    pipe()
    a(f"    {T} mean = 0.0{SF};")
    a("    for (int i = 0; i < n; i++) mean += x[i];")
    a(f"    mean /= ({T})n;")
    a(f"    {T} var = 0.0{SF};")
    a("    for (int i = 0; i < n; i++) {")
    a(f"        {T} d = x[i] - mean; var += d * d;")
    a("    }")
    a(f"    var /= ({T})n;")
    a(f"    {T} inv = 1.0{SF} / sqrt(var + eps);")
    a("    for (int i = 0; i < n; i++) {")
    a("        y[i] = (x[i] - mean) * inv * g[i] + b[i];")
    a("    }")
    a("}")
    a("")
    a(f"static void silu(const {T} *x, int n, {T} *y) {{")
    pipe()
    a("    for (int i = 0; i < n; i++) {")
    a(f"        y[i] = x[i] / (1.0{SF} + exp(-x[i]));")
    a("    }")
    a("}")
    a("")
    a(f"static void gelu(const {T} *x, int n, {T} *y) {{")
    pipe()
    a(f"    const {T} k = 0.7978845608028654{SF};")
    a("    for (int i = 0; i < n; i++) {")
    a(f"        {T} x3 = x[i] * x[i] * x[i];")
    a(f"        {T} u = k * (x[i] + 0.044715{SF} * x3);")
    a("        y[i] = 0.5 * x[i] * (1.0 + tanh(u));")
    a("    }")
    a("}")
    a("")
    a(f"static void rope_apply(const {T} *x, int n, int hd, {T} pos, {T} theta, {T} *y) {{")
    pipe()
    a("    for (int s = 0; s < n; s += hd) {")
    a("        for (int i = 0; i < hd / 2; i++) {")
    a("            int ia = s + i, ib = ia + hd / 2;")
    a(f"            {T} ang = pos / pow(theta, ({T})(2.0 * i / hd));")
    a("            {0} c = cos(ang), sn = sin(ang);".format(T))
    a(f"            {T} xa = x[ia], xb = x[ib];")
    a("            y[ia] = xa * c - xb * sn;")
    a("            y[ib] = xa * sn + xb * c;")
    a("        }")
    a("    }")
    a("}")
    a("")
    a(f"static void softmax1({T} *v) {{")
    pipe()
    a(f"    {T} m = v[0];")
    a("    for (int i = 1; i < HEAD_DIM; i++) if (v[i] > m) m = v[i];")
    a(f"    {T} s = 0.0{SF};")
    a("    for (int i = 0; i < HEAD_DIM; i++) {")
    a("        v[i] = exp(v[i] - m); s += v[i];")
    a("    }")
    a("    for (int i = 0; i < HEAD_DIM; i++) v[i] /= s;")
    a("}")
    a("")
    a(f"/* ---------------- top: one decode step ---------------- */")
    a(f"void {res.top_fname}(int token, int pos, {T} *out) {{")
    # 指针参数只允许 ptrdefault/none/handshake/valid/ovalid/acknowledge/
    # fifo/bus/m_axi/axis（plugin_ASTAnalyzer.cpp ~1412），所以 ap_memory
    # (-> array) 用在 float* 上会报 "Invalid HLS interface mode"。
    # ap_ctrl_hs 同样不在 mode 表里；顶层握手由 --generate-interface=INFER 生成。
    for p in ("#pragma HLS INTERFACE mode=ap_none port=token",
              "#pragma HLS INTERFACE mode=ap_none port=pos",
              "#pragma HLS INTERFACE mode=ap_none port=out"):
        a("    " + p)
    pipe()
    for cname, wdt in alloc.decls():
        a(f"    {T} {cname}[{wdt}];")
    a(f"    for (int i = 0; i < HIDDEN; i++) "
      f"{cmap[emb_prologue[0].outputs[0]]}[i] = {wte_n}[token * HIDDEN + i];")
    if len(emb_prologue) >= 2:
        a(f"    for (int i = 0; i < HIDDEN; i++) "
          f"{cmap[emb_prologue[1].outputs[0]]}[i] = {wpe_n}[pos * HIDDEN + i];")

    final_out = ir.outputs[0] if ir.outputs else (active[-1].outputs[0] if active else "")
    written = set()

    def dst_of(o: str) -> str:
        return "out" if o == final_out else cmap[o]

    for n in active:
        ot = _op_type(n)
        o = n.outputs[0]
        dst = dst_of(o)
        srcs = [cmap[i] for i in n.inputs if i in cmap]
        w_in = width_of.get(n.inputs[0], hidden) if n.inputs else hidden
        widths = [width_of.get(i, hidden) for i in n.inputs if i in cmap]
        if not srcs:
            res.warnings.append(f"{n.name}:{ot} 输入不可达，跳过")
            continue

        if ot == Op.LINEAR.value:
            W = np.asarray(wdata(n.weight_names[0]), dtype=np.float64)
            n_out, n_in = int(W.shape[0]), int(W.shape[1])
            if n_in != w_in:
                raise CKernelError(
                    f"{n.name}: 权重输入维 {n_in} 与张量宽度 {w_in} 不符")
            wn = wconsts.get(n.weight_names[0])
            if wn is None:
                raise CKernelError(f"{n.name}: 权重 {n.weight_names[0]} 未注册")
            a(f"    gemv({wn}, {n_out}, {n_in}, {srcs[0]}, {dst});")
            if len(n.weight_names) > 1:
                b = np.asarray(wdata(n.weight_names[1]), dtype=np.float64).reshape(-1)
                bn = wconsts.get(n.weight_names[1])
                if bn is None:
                    raise CKernelError(
                        f"{n.name}: 偏置 {n.weight_names[1]} 未注册")
                if b.size == n_out:
                    a("    for (int i = 0; i < %d; i++) %s[i] += %s[i];" % (n_out, dst, bn))
                else:
                    res.warnings.append(
                        f"{n.name}: 偏置长度 {b.size} != {n_out}，忽略")
        elif ot == Op.LAYERNORM.value:
            g = np.asarray(wdata(n.weight_names[0]), dtype=np.float64).reshape(-1)
            if len(n.weight_names) > 1:
                bb = np.asarray(wdata(n.weight_names[1]), dtype=np.float64).reshape(-1)
            else:
                bb = np.zeros_like(g)
            gn = wconsts.get(n.weight_names[0])
            bn = wconsts.get(n.weight_names[1]) if len(n.weight_names) > 1 else None
            if gn is None or (len(n.weight_names) > 1 and bn is None):
                raise CKernelError(f"{n.name}: LayerNorm 权重未注册")
            a(f"    layernorm({srcs[0]}, {w_in}, {gn}, {bn}, {_c(eps, LSF)}, {dst});")
        elif ot == Op.RMSNORM.value:
            if n.weight_names:
                gn = wconsts.get(n.weight_names[0])
                if gn is None:
                    raise CKernelError(
                        f"{n.name}: gamma {n.weight_names[0]} 未注册")
            else:
                gn = ones_n
            a(f"    rmsnorm({srcs[0]}, {w_in}, {gn}, {_c(eps, LSF)}, {dst});")
        elif ot == Op.ROPE.value:
            a(f"    rope_apply({srcs[0]}, {w_in}, HEAD_DIM, ({T})pos, "
              f"{_c(rope_theta, LSF)}, {dst});")
        elif ot == Op.ATTENTION.value:
            vsrc = None
            for i in n.inputs[2:]:
                if i in cmap:
                    vsrc = cmap[i]
                    break
            if vsrc is None:
                vsrc = srcs[-1]
            a(f"    for (int i = 0; i < {w_in}; i++) {dst}[i] = {vsrc}[i];")
            res.warnings.append(
                f"{n.name}: attention 在单步 decode(KV 长度=1) 下退化为 v 透传")
        elif ot in (Op.SILU.value, Op.GELU.value):
            fn = "silu" if ot == Op.SILU.value else "gelu"
            a(f"    {fn}({srcs[0]}, {w_in}, {dst});")
        elif ot == Op.SOFTMAX.value:
            a(f"    for (int i = 0; i < {w_in}; i++) {dst}[i] = {srcs[0]}[i];")
            a(f"    softmax1({dst});")
        elif ot in (Op.ADD.value, Op.MUL.value, Op.SUB.value):
            m = max(widths) if len(widths) > 1 else w_in
            if len(srcs) >= 2:
                sym = {Op.ADD.value: "+", Op.MUL.value: "*", Op.SUB.value: "-"}[ot]
                a(f"    for (int i = 0; i < {m}; i++) "
                  f"{dst}[i] = {srcs[0]}[i] {sym} {srcs[1]}[i];")
            else:
                a(f"    for (int i = 0; i < {m}; i++) {dst}[i] = {srcs[0]}[i];")
        elif ot == Op.RELU.value:
            a(f"    for (int i = 0; i < {w_in}; i++) {dst}[i] = "
              f"{srcs[0]}[i] > 0 ? {srcs[0]}[i] : 0;")
        else:
            res.warnings.append(f"{n.name}:{ot} 未映射，按恒等处理")
            a(f"    for (int i = 0; i < {w_in}; i++) {dst}[i] = {srcs[0]}[i];")
        written.add(o)
        ow = width_of.get(o, w_in)
        res.trace.append((n.name, ot, list(srcs), dst, ow))
        if kcfg.trace:
            nl = "\\n"
            a(f'    printf("TRACE {n.name} {ot} w={ow}{nl}");')
            a(f"    for (int i = 0; i < {ow}; i++) "
              f'printf("%d %.17g{nl}", i, (double){dst}[i]);')

    if final_out and final_out in cmap and final_out not in written:
        a(f"    for (int i = 0; i < VOCAB; i++) out[i] = {cmap[final_out]}[i];")
    elif final_out and final_out not in written and final_out in width_of:
        a(f"    for (int i = 0; i < VOCAB; i++) out[i] = 0.0{SF};")

    a("}")
    a("")
    if kcfg.emit_main:
        a("#ifndef SYNTHESIS")
        a("int main(int argc, char **argv) {")
        a("    int token = (argc > 1) ? atoi(argv[1]) : 0;")
        a("    int pos = (argc > 2) ? atoi(argv[2]) : 0;")
        a(f"    {T} out[VOCAB];")
        a(f"    for (int i = 0; i < VOCAB; i++) out[i] = 0.0{SF};")
        a(f"    {res.top_fname}(token, pos, out);")
        a('    for (int i = 0; i < VOCAB; i++) printf("%.9g\\n", (double)out[i]);')
        a("    return 0;")
        a("}")
        a("#endif")

    body = "\n".join(L) + "\n"
    top_path = os.path.join(out_dir, f"{kcfg.prefix}_kernel.cpp")
    with open(top_path, "w", encoding="utf-8") as f:
        f.write(body)
    res.top_path = top_path
    res.line_count = body.count("\n")
    return res
