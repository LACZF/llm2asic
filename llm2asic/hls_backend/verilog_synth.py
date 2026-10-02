# llm2asic/hls_backend/verilog_synth.py
"""把第三方 HLS 工具产出的 Verilog 里的**仿真专用**构造改写成可综合写法。

Bambu (PandA) 生成的每个存储体模板都带这么一段：

    initial
    begin
      $readmemb(MEMORY_INIT_file, memory, 0, n_elements-1);
    end

`initial` + `$readmemb` 只在仿真里有意义，综合器不会（也不应该）去读一个
外部 .mem 文件：ASIC 流程（DC/Genus/OpenROAD）直接报语法错，很多 FPGA 流程
在 strict 模式下也报错，不 strict 就**静默把初值丢掉**——权重全 0，网表是错的。

所以这里把它换成常量 ``case`` ROM：纯组合逻辑，任何工具都能综合，
而且逐位等价于 ``$readmemb`` 读出来的初值。

约定（见 :func:`make_rom_synthesizable` 里的注释）：

* ``READ_ONLY_MEMORY=1`` 的存储体**没有任何写路径**（Bambu 自己用
  ``generate if (READ_ONLY_MEMORY==0)`` 把写逻辑关掉了），所以读端直接换成
  case ROM，原数组没人写也没人读，综合时会被优化掉。
* ``READ_ONLY_MEMORY=0`` 的存储体运行时会被 FSM 覆写，它的 ``initial``
  内容在 Bambu 自己的仿真里就是不确定的（实测 ``.mem`` 只有 4 个字而
  ``n_elements=8``，一半是 x）。这类只删 ``initial``，读端仍走数组。
"""

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "RomRewrite",
    "scan_simulation_only",
    "make_rom_synthesizable",
]


# 综合器不接受的构造。`initial` 只在没有别的内容时才报（工具自己生成的
# 寄存器初值偶尔是合法的），所以单独给一条"仅 initial"的检查。
SIMULATION_ONLY = {
    "readmem": r"\$readmem[hb]?\s*\(",
    "display": r"\$(?:display|write|strobe|monitor)\b",
    "fileio": r"\$(?:fopen|fclose|fwrite|fdisplay|fscanf|fmonitor|fflush)\b",
    "finish": r"\$(?:finish|stop)\b",
    "severity": r"\$(?:fatal|error|warning|info)\b",
    "dumpvars": r"\$(?:dumpfile|dumpvars|dumpall|dumplimit|monitoron)\b",
    "time": r"\$(?:time|stime|realtime)\b",
    "random": r"\$(?:random|urandom|urandom_range|dist_)\b",
    "assert": r"\$(?:assert|assertkill|assertoff|asserton)\b",
    "sformat": r"\$(?:sformat|sformatf|swrite|sscanf)\b",
    "delay": r"#[ \t]*[0-9_]+[ \t]*(?:;|\)|,|$)",
    "specify": r"^[ \t]*specify\b",
    "force": r"\b(?:force|release|deassign)\b",
}

_ONLY_INITIAL = {"initial": r"\binitial\b"}


def scan_simulation_only(text: str, strict_initial: bool = True) -> Dict[str, int]:
    """统计文本里还剩多少仿真专用构造。

    Parameters
    ----------
    strict_initial :
        True 时单独统计裸 ``initial``（可能是仿真初始化块）；
        False 时忽略 initial，只看真正不可综合的系统任务。

    Returns
    -------
    dict
        ``{构造名: 出现次数}``，为空表示没发现问题。
    """
    pats = dict(SIMULATION_ONLY)
    if strict_initial:
        pats.update(_ONLY_INITIAL)
    found = {}
    for name, pat in pats.items():
        n = len(re.findall(pat, text, re.MULTILINE))
        if n:
            found[name] = n
    return found


# ---------------------------------------------------------------- 解析

_MODULE_RE = re.compile(r"^module\s+(\w+)", re.MULTILINE)
_INITIAL_RE = re.compile(r"\binitial\b")
_TOKEN_RE = re.compile(r"\b(begin|end)\b|\$readmem[hb]?\s*\(|//[^\n]*",
                       re.IGNORECASE)
_READMEM_CALL_RE = re.compile(
    r"\$readmem([hb]?)\s*\(\s*([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*)\s*,")
_ARRAY_DECL_RE = re.compile(
    r"^[ \t]*reg\s*(\[[^\]\n]*\])\s*([A-Za-z_]\w*)\s*(\[[^\]\n]*\])",
    re.MULTILINE)
_PARAM_STR_RE = r'\.{p}\s*\(\s*"([^"]*)"\s*\)'
_N_ELEMENTS_RE = r"\.n_elements\s*\(\s*(\d+)\s*\)"

_ROM_FN = "llm2asic_rom_{arr}"
_ROM_ADDR_W = 32


@dataclass
class RomRewrite:
    """一次改写的结果。"""

    path: str = ""
    modules: List[str] = field(default_factory=list)
    files_baked: List[str] = field(default_factory=list)
    files_missing: List[str] = field(default_factory=list)
    readmem_removed: int = 0
    reads_romified: int = 0
    warnings: List[str] = field(default_factory=list)
    changed: bool = False

    def summary(self) -> str:
        if not self.changed:
            return "未发现 initial/$readmemb，无需改写"
        return (f"已把 {self.readmem_removed} 处 initial/$readmemb 改写为可综合 "
                f"case ROM（{len(self.modules)} 个存储体模板，固化 "
                f"{len(self.files_baked)} 个 .mem，{self.reads_romified} 处读端"
                f"改为查表）")


def _split_modules(text: str) -> List[Tuple[str, int, int]]:
    """返回 [(模块名, 起点, 终点)]，终点是 ``endmodule`` 之后的位置。"""
    starts = [(m.start(), m.group(1)) for m in _MODULE_RE.finditer(text)]
    out = []
    for i, (off, name) in enumerate(starts):
        e = text.find("endmodule", off)
        if e < 0:
            end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
        else:
            end = text.find("\n", e)
            end = len(text) if end < 0 else end + 1
        out.append((name, off, end))
    return out


def _read_mem_words(path: str, mode: str = "b") -> List[Tuple[int, int]]:
    """读 ``.mem``，返回 ``[(值, 位宽), ...]``。

    必须按 ``mode`` 解释，**不能一律当十六进制**：

    * ``$readmemb``（Bambu 用的就是这个）：一个字符 = **一个 bit**。
      实测 gpt2_tiny 的 token 全是 32 个 ``0``/``1``，那是 32 bit 的值，
      按 hex 读会变成 128 bit，权重直接错掉。
    * ``$readmemh``：一个字符 = 4 bit。

    ``x``/``z`` 按 0 处理（对应"该地址没被 $readmem 覆盖"，综合后同样是
    未初始化，换成确定的 0 更好）。
    """
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            raw = f.read()
    except OSError:
        return []
    base = 16 if mode.lower() == "h" else 2
    words = []
    for tok in raw.split():
        if tok.startswith("//") or tok.startswith("#"):
            continue
        tok = re.sub(r"[xXzZ]", "0", tok)
        try:
            value = int(tok, base)
        except ValueError:
            continue
        width = len(tok) * (4 if base == 16 else 1)
        words.append((value, max(width, 1)))
    return words


def _hex_literal(value: int, width: int) -> str:
    return f"{width}'h{value:x}"


def _strip_initial_readmem(body: str) -> Tuple[str, int]:
    """删掉含 ``$readmem`` 的 ``initial`` 块（成对匹配 begin/end）。

    不能用 ``.*?end`` 贪到第一个 end：Bambu 的块长这样::

        initial
        begin
          if (MEMORY_INIT_file != "")
            $readmemb(F, memory, 0, n_elements-1);
          else
          begin
            for(index=0; ...) begin memory[index] = 0; end
          end
        end

    非贪婪匹配会停在 for 体的 ``end``，留下一堆孤儿 ``end`` 直接语法错。
    """
    out = []
    pos = 0
    removed = 0
    while True:
        m = _INITIAL_RE.search(body, pos)
        if m is None:
            out.append(body[pos:])
            break
        # 先扫出这个 initial 块的完整范围
        depth = 0
        end = None
        has_readmem = False
        for t in _TOKEN_RE.finditer(body, m.end()):
            tok = t.group(0)
            low = tok.lower()
            if low.startswith("//"):
                continue
            if low.startswith("$readmem"):
                has_readmem = True
                continue
            if low == "begin":
                depth += 1
            elif low == "end":
                depth -= 1
                if depth <= 0:
                    end = t.end()
                    break
        if end is None:
            out.append(body[pos:])
            break
        if has_readmem:
            # 连带吃掉这一行前面的缩进/换行
            line_start = body.rfind("\n", 0, m.start()) + 1
            if not body[line_start:m.start()].strip():
                m_start = line_start
            else:
                m_start = m.start()
            nxt = body.find("\n", end)
            nxt = len(body) if nxt < 0 else nxt + 1
            out.append(body[pos:m_start])
            pos = nxt
            removed += 1
        else:
            out.append(body[pos:end])
            pos = end
    return "".join(out), removed


def _decl_range(decl_text: str) -> Tuple[str, str]:
    """把声明里的 ``[msb:lsb]`` 拆成 (msb 表达式, lsb 表达式)。"""
    inner = decl_text.strip().lstrip("[").rstrip("]")
    parts = inner.split(":")
    if len(parts) != 2:
        return inner, "0"
    return parts[0].strip(), parts[1].strip()


def _classify_reads(body: str, arr: str, skip: Sequence[Tuple[int, int]] = ()):
    """找出 ``arr[...]`` 的引用，区分写（LHS）和读（RHS）。

    ``skip`` 里的区间会被忽略——数组**声明**本身也是 ``arr[...]``，
    当成读就会把声明改坏。

    Returns
    -------
    list[(start, end, addr_expr, is_write)]
        ``[start, end)`` 是 ``arr[addr]`` 整段在 body 里的区间。
    """
    out = []
    for m in re.finditer(rf"\b{re.escape(arr)}\s*\[", body):
        if any(a <= m.start() < b for a, b in skip):
            continue
        i = m.end() - 1                      # 指向 '['
        depth = 0
        j = i
        while j < len(body):
            if body[j] == "[":
                depth += 1
            elif body[j] == "]":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if j >= len(body):
            continue
        addr = body[i + 1:j]
        k = j + 1
        # 字节使能会写子域：memory[a][i*8+:8] <= ...。跟着吃掉所有
        # 尾随的 []，否则会把这个写操作误判成读。
        while True:
            t = k
            while t < len(body) and body[t] in " \t\r\n":
                t += 1
            if t < len(body) and body[t] == "[":
                depth = 0
                u = t
                while u < len(body):
                    if body[u] == "[":
                        depth += 1
                    elif body[u] == "]":
                        depth -= 1
                        if depth == 0:
                            break
                    u += 1
                if u >= len(body):
                    break
                k = u + 1
                continue
            k = t
            break
        is_write = body[k:k + 2] == "<=" or (body[k:k + 1] == "="
                                            and body[k + 1:k + 2] != "=")
        out.append((m.start(), j + 1, addr, is_write))
    return out


def _collect_param_files(text: str, module: str, param: str) -> Tuple[List[str], Dict[str, int]]:
    """某模块的 ``param`` 可能取哪些文件名，以及每个文件对应的 n_elements。

    来源：所有 ``module #(...) inst(...)`` 例化上的 ``.param("x.mem")`` 覆盖，
    加上模块自身的默认值。只把**例化里真正用到的**文件烤进 ROM，
    免得给一个模板塞进几十个无关的 case。
    """
    files: List[str] = []
    depths: Dict[str, int] = {}
    for m in re.finditer(rf"\b{re.escape(module)}\s*#\s*\(", text):
        i = m.end()
        depth = 1
        j = i
        while depth and j < len(text):
            if text[j] == "(":
                depth += 1
            elif text[j] == ")":
                depth -= 1
            j += 1
        params = text[i:j - 1]
        name = re.search(_PARAM_STR_RE.format(p=param), params)
        if not name:
            continue
        fn = name.group(1)
        if fn not in files:
            files.append(fn)
        nd = re.search(_N_ELEMENTS_RE, params)
        if nd:
            depths[fn] = max(depths.get(fn, 0), int(nd.group(1)))
    return files, depths


def _module_param_default(body: str, param: str) -> Optional[str]:
    m = re.search(rf"\b{re.escape(param)}\s*=\s*\"([^\"]*)\"", body)
    return m.group(1) if m else None


def _build_rom_function(arr: str, msb: str, lsb: str, param: str,
                        files: Sequence[str], words_of) -> str:
    # 复制次数必须是常量表达式；msb/lsb 都是模块参数，拼出来合法。
    z = f"{{(({msb})-({lsb})+1){{1'b0}}}}"
    parts = [f"  function [{msb}:{lsb}] {_ROM_FN.format(arr=arr)};",
             f"    input [{_ROM_ADDR_W - 1}:0] llm2asic_rom_addr;",
             "    begin",
             f"      {_ROM_FN.format(arr=arr)} = {z};"]
    for fn in files:
        words = words_of(fn)
        if not words:
            continue
        parts.append(f'      if ({param} == "{fn}") begin')
        parts.append("        case (llm2asic_rom_addr)")
        for addr, (val, w) in enumerate(words):
            parts.append(f"          {addr}: {_ROM_FN.format(arr=arr)} = "
                         f"{_hex_literal(val, w)};")
        parts.append("        endcase")
        parts.append("      end")
    parts.append("    end")
    parts.append("  endfunction")
    return "\n".join(parts)


def make_rom_synthesizable(verilog_path: str,
                           search_dirs: Sequence[str]) -> RomRewrite:
    """就地改写 ``verilog_path``，把 ``initial/$readmemb`` 换成 case ROM。

    Parameters
    ----------
    verilog_path :
        要改写的 .v（原地修改）。
    search_dirs :
        找 ``.mem`` 的目录，按顺序取第一个存在的。

    Returns
    -------
    RomRewrite
        改写统计 + 告警。找不到任何 ``$readmem`` 时 ``changed=False``。
    """
    res = RomRewrite(path=verilog_path)
    try:
        with open(verilog_path, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError as e:
        res.warnings.append(f"读不到 Verilog，无法改写: {e}")
        return res

    if "$readmem" not in text:
        return res

    def find_mem(name):
        for d in search_dirs:
            p = name if os.path.isabs(name) else os.path.join(d, name)
            if os.path.isfile(p):
                return p
        return None

    cache: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}

    def words_of(name, mode="b"):
        key = (name, mode)
        if key not in cache:
            p = find_mem(name)
            if p is None:
                cache[key] = []
                if name not in res.files_missing:
                    res.files_missing.append(name)
            else:
                cache[key] = _read_mem_words(p, mode)
                # 记解析后的真实路径：下游要能确认文件确实在
                if p not in res.files_baked:
                    res.files_baked.append(p)
        return cache[key]

    # 从后往前改，前面模块的偏移才不会失效
    for name, off, end in reversed(_split_modules(text)):
        body = text[off:end]
        if "$readmem" not in body:
            continue

        calls = _READMEM_CALL_RE.findall(body)
        if not calls:
            continue
        rom_of = {}
        inserts: List[Tuple[int, str]] = []      # (相对 body 的位置, 文本)
        replacements: List[Tuple[int, int, str]] = []
        gate_param = ("READ_ONLY_MEMORY"
                      if re.search(r"\bREAD_ONLY_MEMORY\b", body) else None)

        for mode, param, arr in calls:
            decl = None
            for d in _ARRAY_DECL_RE.finditer(body):
                if d.group(2) == arr:
                    decl = d
                    break
            if decl is None:
                res.warnings.append(
                    f"{name}: 找不到数组 {arr} 的声明，跳过 ROM 改写")
                continue
            msb, lsb = _decl_range(decl.group(1))
            rom_of[arr] = _ROM_FN.format(arr=arr)

            files, depths = _collect_param_files(text, name, param)
            default = _module_param_default(body, param)
            if default and default not in files:
                files.append(default)
            if not files:
                res.warnings.append(
                    f"{name}: 找不到 {param} 的取值，{arr} 保持数组读")
                continue

            for fn in files:
                ws = words_of(fn, mode)
                if not ws:
                    continue
                limit = depths.get(fn)
                if limit is not None and len(ws) > limit:
                    res.warnings.append(
                        f"{name}.{arr}: {fn} 有 {len(ws)} 个字但 "
                        f"n_elements={limit}，多余的字综合后不会被选中")

            reads = _classify_reads(body, arr,
                                    skip=[(decl.start(), decl.end())])
            if gate_param is None:
                # 没有只读开关：这个数组一定会被写，ROM 用不上
                continue
            for start, stop, addr, is_write in reads:
                if is_write:
                    continue
                replacements.append(
                    (start, stop,
                     f"({gate_param} ? {rom_of[arr]}({addr}) : {arr}[{addr}])"))
                res.reads_romified += 1

            # 函数必须出现在第一次调用之前。插到**声明之前**：声明后面还
            # 跟着注释和分号（reg [..] mem [..] /* syn_ramstyle */ ;），
            # 插在 decl.end() 会把声明截断成语法错误。
            first_use = min((st for st, _e, _a, w in reads if not w),
                            default=len(body))
            head = body[:decl.start()]
            if len(re.findall(r"\bgenerate\b", head)) != \
                    len(re.findall(r"\bendgenerate\b", head)):
                res.warnings.append(
                    f"{name}: {arr} 声明在 generate 块里，Verilog-2001 不允许"
                    f"块内声明 function，跳过 ROM 改写")
                continue
            inserts.append((min(decl.start(), first_use),
                            _build_rom_function(arr, msb, lsb, param, files,
                                                lambda n, _m=mode: words_of(n, _m))))

        for start, stop, new in sorted(replacements, reverse=True):
            body = body[:start] + new + body[stop:]
        for pos, txt in sorted(inserts, reverse=True):
            body = body[:pos] + "\n" + txt + "\n" + body[pos:]

        body, n = _strip_initial_readmem(body)
        res.readmem_removed += n
        if n:
            res.modules.append(name)
        text = text[:off] + body + text[end:]

    if res.readmem_removed or res.reads_romified:
        try:
            with open(verilog_path, "w", encoding="utf-8") as f:
                f.write(text)
            res.changed = True
        except OSError as e:
            res.warnings.append(f"改写失败: {e}")
            res.changed = False

    try:
        with open(verilog_path, encoding="utf-8", errors="replace") as f:
            leftover = "$readmem" in f.read()
    except OSError:
        leftover = False
    if leftover:
        res.warnings.append("仍有 $readmem 残留，请检查存储体模板是否被识别")
    # 找不到 .mem -> 该存储体按全零 ROM 综合。这是最危险的静默行为
    # （权重全零，Verilog 照样过 check），必须报出来而不是默默烤零。
    if res.files_missing:
        res.warnings.append(
            f"{len(res.files_missing)} 个初始化文件找不到，"
            f"对应存储体按全零 ROM 综合: " + ", ".join(res.files_missing))
    return res
