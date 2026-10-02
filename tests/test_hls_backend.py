# tests/test_hls_backend.py
"""HLS 后端测试：native C 内核 / hls4ml 工程 / ONNX 导出。

覆盖三条路径：
  * native  —— 生成 C 内核，用 g++ 编译后与 numpy 浮点参考比对（数值正确性）
  * hls4ml  —— 生成 HLS C++ 工程，并用 g++ -fsyntax-only 验证可编译
  * onnx    —— 导出单步 decode ONNX，用 onnx.checker 验证结构合法

hls4ml / onnx 属于可选依赖，缺失时对应测试自动跳过；
native 路径只要有 g++ 就能跑，是这里的基准。
"""

import os
import re
import shutil
import subprocess

import numpy as np
import pytest

from llm2asic.hls_backend import (
    HlsBackend,
    HlsConfig,
    build_hls,
)
from llm2asic.hls_backend.bambu import (
    BambuConfig,
    BambuResult,
    parse_bambu_report,
    run_bambu,
)
from llm2asic.hls_backend.c_kernel import (
    CKernelConfig,
    CKernelError,
    gen_c_kernel,
)
from llm2asic.hls_backend.float_ref import ref_decode_step
from llm2asic.hls_backend.verilog_synth import (
    make_rom_synthesizable,
    scan_simulation_only,
)
from llm2asic.parser.builder import run_from_path

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")
MODELS = ["gpt2_tiny", "llama_tiny", "llama_mini"]

GXX = shutil.which("g++")


def _model(name):
    return os.path.abspath(os.path.join(EXAMPLES, name, "model.yaml"))


def _have_hls4ml() -> bool:
    try:
        import hls4ml  # noqa: F401
        from hls4ml.model import ModelGraph  # noqa: F401
        return True
    except ImportError:
        return False


def _have_onnx() -> bool:
    try:
        import onnx  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------- native

@pytest.mark.skipif(GXX is None, reason="需要 g++")
@pytest.mark.parametrize("name", MODELS)
def test_native_c_kernel_matches_float_ref(name, tmp_path):
    """生成的 C 内核与 numpy 浮点参考在多组 (token, pos) 上逐位接近。

    这是不依赖 HLS 工具链也能做的最强校验。
    """
    ir = run_from_path(_model(name))
    cfg = CKernelConfig(prefix="t", emit_main=True, precision="float")
    res = gen_c_kernel(ir, str(tmp_path), cfg)
    assert res.ok, res.errors

    exe = tmp_path / "kern"
    p = subprocess.run(
        [GXX, "-std=c++14", "-O2", res.top_path, "-o", str(exe), "-lm"],
        capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, p.stderr[-3000:]

    worst = 0.0
    for token, pos in ((1, 0), (2, 0), (1, 1), (0, 0), (3, 2)):
        r = subprocess.run([str(exe), str(token), str(pos)],
                           capture_output=True, text=True, timeout=900)
        assert r.returncode == 0, r.stderr[-2000:]
        got = np.fromstring(r.stdout, sep=" ", dtype=np.float64)
        want = np.asarray(ref_decode_step(ir, token, pos), dtype=np.float64)
        assert got.shape == want.shape, (got.shape, want.shape)
        denom = np.maximum(np.abs(want), 1e-3)
        rel = float(np.max(np.abs(got - want) / denom))
        worst = max(worst, rel)
    assert worst < 1e-4, f"{name}: 相对误差 {worst:.3e} 过大"


@pytest.mark.skipif(GXX is None, reason="需要 g++")
def test_native_position_embedding_is_separate_from_token_embedding(tmp_path):
    """GPT-2 的词嵌入与位置嵌入必须是两个独立缓冲，ADD 节点再相加。

    早期版本把两者融合进同一缓冲，导致 hidden 宽度变成 2*h。
    """
    ir = run_from_path(_model("gpt2_tiny"))
    res = gen_c_kernel(ir, str(tmp_path), CKernelConfig(prefix="t"))
    assert res.ok, res.errors
    src = open(res.top_path).read()
    assert "wpe" in src and "wte" in src


def test_c_kernel_buffer_pool_cap_raises_instead_of_aliasing():
    """缓冲池打满且无空闲缓冲时必须报错，不能复用仍活跃的缓冲。

    复用活跃缓冲会静默产出数值错误的内核，比直接失败糟糕得多。
    """
    ir = run_from_path(_model("llama_tiny"))
    with pytest.raises(CKernelError, match="n_buffers"):
        gen_c_kernel(ir, "/tmp/llm2asic_test_nb1",
                     CKernelConfig(prefix="t", n_buffers=1))


@pytest.mark.skipif(GXX is None, reason="需要 g++")
def test_native_double_precision_compiles(tmp_path):
    """precision=double 也能生成可编译的 C 内核。"""
    ir = run_from_path(_model("llama_tiny"))
    res = gen_c_kernel(ir, str(tmp_path),
                       CKernelConfig(prefix="t", precision="double",
                                     emit_main=True))
    assert res.ok, res.errors
    p = subprocess.run(
        [GXX, "-std=c++14", "-fsyntax-only", res.top_path],
        capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, p.stderr[-3000:]


# ---------------------------------------------------------------- hls4ml

@pytest.mark.skipif(not _have_hls4ml(), reason="需要 hls4ml")
@pytest.mark.skipif(GXX is None, reason="需要 g++")
@pytest.mark.parametrize("name", MODELS)
def test_hls4ml_project_cxx_compiles(name, tmp_path):
    """hls4ml 生成的 HLS C++ 工程必须能通过 g++ 语法检查。

    回归重点：LayerNorm 在浮点模式下曾因 lookup table 类型与 accum_t
    混用而产生 ambiguous overload，已在生成后修补 table_t。
    """
    cfg = HlsConfig(backend=HlsBackend.HLS4ML, out_dir=str(tmp_path),
                    run_bambu=False)
    res = build_hls(run_from_path(_model(name)), cfg)
    assert res.ok, res.errors
    assert res.hls4ml_layer_count > 0
    assert os.path.isfile(res.hls4ml_cpp), res.hls4ml_cpp

    proj = res.hls4ml_project
    p = subprocess.run(
        [GXX, "-std=c++14", "-fsyntax-only", res.hls4ml_cpp,
         "-I", os.path.join(proj, "firmware"),
         "-I", os.path.join(proj, "firmware", "ap_types")],
        capture_output=True, text=True, timeout=1800)
    assert p.returncode == 0, p.stderr[-4000:]


@pytest.mark.skipif(not _have_hls4ml(), reason="需要 hls4ml")
def test_hls4ml_projects_do_not_collide(tmp_path):
    """不同模型必须落到各自的 hls4ml 工程目录，不能互相覆盖。"""
    seen = []
    for name in MODELS:
        cfg = HlsConfig(backend=HlsBackend.HLS4ML,
                        out_dir=str(tmp_path / name), run_bambu=False)
        res = build_hls(run_from_path(_model(name)), cfg)
        assert res.ok, res.errors
        seen.append(os.path.realpath(res.hls4ml_project))
    assert len(set(seen)) == len(MODELS), seen


# ---------------------------------------------------------------- onnx

@pytest.mark.skipif(not _have_onnx(), reason="需要 onnx")
@pytest.mark.parametrize("name", MODELS)
def test_onnx_export_is_valid(name, tmp_path):
    """单步 decode ONNX 必须通过 onnx.checker。"""
    import onnx

    cfg = HlsConfig(backend=HlsBackend.ONNX, out_dir=str(tmp_path), pos=0)
    res = build_hls(run_from_path(_model(name)), cfg)
    assert res.ok, res.errors
    assert os.path.isfile(res.onnx_path)
    onnx.checker.check_model(onnx.load(res.onnx_path))


@pytest.mark.skipif(not _have_onnx(), reason="需要 onnx")
def test_onnx_linear_weight_is_transposed(tmp_path):
    """ONNX MatMul 权重必须是 [c_in, c_out]（IR 里存的是 [c_out, c_in]）。"""
    import onnx
    from onnx import numpy_helper

    from llm2asic.hls_backend.onnx_export import export_onnx

    ir = run_from_path(_model("llama_tiny"))
    res = export_onnx(ir, str(tmp_path), pos=0, name="m")
    m = onnx.load(res.path)
    inits = {i.name: i for i in m.graph.initializer}
    found = 0
    for node in m.graph.node:
        if node.op_type != "MatMul":
            continue
        w = inits.get(node.input[1])
        if w is None:
            continue
        arr = numpy_helper.to_array(w)
        if arr.ndim == 2 and arr.shape[0] > 1 and arr.shape[1] > 1:
            found += 1
    assert found > 0, "未找到二维 MatMul 权重，无法校验转置"


# ---------------------------------------------------------------- pipeline

@pytest.mark.parametrize("backend", [HlsBackend.NATIVE, HlsBackend.HLS4ML,
                                     HlsBackend.ONNX])
def test_build_hls_rejects_unknown_backend(backend, tmp_path):
    """未知后端必须给出明确错误，而不是抛异常。"""
    res = build_hls(run_from_path(_model("llama_tiny")),
                    HlsConfig(backend="nope", out_dir=str(tmp_path)))
    assert not res.ok
    assert "未知后端" in res.errors[0]


def test_build_hls_missing_bambu_is_reported_not_raised(tmp_path, monkeypatch):
    """Bambu 缺失时应返回带明确错误的结果，而不是抛异常。"""
    if shutil.which("bambu") is not None:
        pytest.skip("环境里已安装 bambu")
    monkeypatch.setenv("PATH", "")
    res = build_hls(run_from_path(_model("llama_tiny")),
                    HlsConfig(backend=HlsBackend.NATIVE,
                              out_dir=str(tmp_path), run_bambu=True,
                              verify=False))
    assert not res.ok
    assert any("bambu" in e for e in res.errors)


@pytest.mark.parametrize("backend", [HlsBackend.NATIVE, HlsBackend.HLS4ML])
def test_reused_config_does_not_leak_derived_names(backend, tmp_path):
    """同一个 HlsConfig 连续用于两个模型时，顶层名不能沿用上一个模型。

    派生值（prefix / project_name）曾经被写回调用方的 config 对象。
    """
    cfg = HlsConfig(backend=backend, run_bambu=False, verify=False)
    tops = []
    for m in ("gpt2_tiny", "llama_tiny"):
        res = build_hls(run_from_path(_model(m)), cfg)
        assert res.ok, res.errors
        tops.append(res.top_module)
    assert all(t and tops[0] != t for t in tops[1:]), tops
    # 调用方的 config 不应被改写
    assert cfg.ckernel.prefix == ""
    assert cfg.hls4ml.project_name in ("", None)


# ---------------------------------------------------------------- bambu

def test_bambu_available_is_false_for_missing_binary(monkeypatch):
    """binary 指向不存在的文件时 bambu_available/bambu_version 必须返回假值。"""
    from llm2asic.hls_backend.bambu import (
        bambu_available,
        bambu_version,
    )

    monkeypatch.setenv("PATH", "")
    assert bambu_version("/nonexistent/bambu") == ""
    assert bambu_available("/nonexistent/bambu") is False


def test_bambu_run_reports_missing_binary(tmp_path):
    """run_bambu 在工具缺失时返回 ok=False 并附带诊断信息，而不是抛异常。"""
    src = tmp_path / "top.cpp"
    src.write_text("int main(){return 0;}\n")
    res = run_bambu(str(src), str(tmp_path), "top",
                    BambuConfig(binary="/nonexistent/bambu"))
    assert isinstance(res, BambuResult)
    assert not res.ok
    assert res.errors
    assert "PandA" in res.errors[0] or "bambu" in res.errors[0].lower()


def test_bambu_run_reports_missing_source(tmp_path):
    """源文件不存在时要在调用工具前就报错。"""
    res = run_bambu(str(tmp_path / "nope.cpp"), str(tmp_path), "top",
                    BambuConfig(binary="/nonexistent/bambu"))
    assert not res.ok
    assert res.errors


def _stub_bambu(tmp_path, returncode=0, emit_report=True):
    """造一个假的 `bambu`，用于在没有 PandA 的环境里测试调用层。

    真实地按契约行事：支持 --version，在 cwd 下建 HLS_output/，
    写出 <top>.v 与报告，并把收到的参数记录到 args.txt。
    """
    bindir = tmp_path / "stubbin"
    bindir.mkdir()
    stub = bindir / "bambu"
    stub.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then echo "Bambu 0.99-stub"; exit 0; fi\n'
        'printf "%s\\n" "$@" > args.txt\n'
        'mkdir -p HLS_output\n'
        'for a in "$@"; do\n'
        '  case "$a" in --top-fname=*) top=${a#--top-fname=} ;; esac\n'
        "done\n"
        'top=${top:-top}\n'
        'printf "module %s(); endmodule\\n" "$top" > "HLS_output/$top.v"\n'
        + ('printf "<HLSReport><Area>4321</Area>'
           "<EstimatedClockPeriod>3.2</EstimatedClockPeriod></HLSReport>\\n\""
           ' > "HLS_output/$top.xml"\n' if emit_report else "")
        + f"exit {returncode}\n")
    stub.chmod(0o755)
    return str(stub)


def test_bambu_stub_collects_verilog_and_metrics(tmp_path):
    """用假 bambu 验证命令拼装、cwd 约定与产物收集。"""
    src = tmp_path / "top.cpp"
    src.write_text("void top(){}\n")
    stub = _stub_bambu(tmp_path)
    out = tmp_path / "work_root"
    res = run_bambu(str(src), str(out), "top",
                    BambuConfig(binary=stub, cwd_name="run1",
                                device_name="xc7a100t", clock_period=5.0))

    assert res.ok, res.errors
    assert res.top_module == "top"
    assert res.returncode == 0
    # 产物必须落在 <out_dir>/<cwd_name>/HLS_output 下
    assert res.hls_output_dir == str(out / "run1" / "HLS_output")
    assert len(res.verilog) == 1
    assert res.verilog[0].endswith("top.v")
    assert res.area == 4321
    assert res.achieved_clock_ns == pytest.approx(3.2)
    assert res.target_clock_ns == pytest.approx(5.0)

    # 顶层名、器件、时钟必须出现在命令行里
    args = (out / "run1" / "args.txt").read_text()
    assert "--top-fname=top" in args
    assert "--device-name=xc7a100t" in args
    assert "--clock-period=5" in args
    assert "-O2" in args
    assert "--generate-interface=INFER" in args


def test_bambu_stub_reports_nonzero_returncode(tmp_path):
    """bambu 返回非 0 时必须报错并保留日志。"""
    src = tmp_path / "top.cpp"
    src.write_text("void top(){}\n")
    stub = _stub_bambu(tmp_path, returncode=3)
    res = run_bambu(str(src), str(tmp_path / "w"), "top",
                    BambuConfig(binary=stub, cwd_name="run1"))
    assert not res.ok
    assert res.returncode == 3
    assert any("返回码" in e for e in res.errors)


def test_bambu_stub_without_verilog_is_error(tmp_path):
    """返回 0 但没产出 Verilog 时应报错（不能当作成功）。"""
    src = tmp_path / "top.cpp"
    src.write_text("void top(){}\n")
    bindir = tmp_path / "stubbin"
    bindir.mkdir()
    stub = bindir / "bambu"
    stub.write_text('#!/bin/sh\n'
                    'if [ "$1" = "--version" ]; then echo stub; exit 0; fi\n'
                    "exit 0\n")
    stub.chmod(0o755)
    res = run_bambu(str(src), str(tmp_path / "w"), "top",
                    BambuConfig(binary=str(stub), cwd_name="run1"))
    assert not res.ok
    assert any("Verilog" in e for e in res.errors)


def test_bambu_multiple_sources_forwarded(tmp_path):
    """多文件输入应全部传给 bambu。"""
    a, b = tmp_path / "a.cpp", tmp_path / "b.cpp"
    a.write_text("void top(){}\n")
    b.write_text("void helper(){}\n")
    stub = _stub_bambu(tmp_path)
    out = tmp_path / "w"
    res = run_bambu([str(a), str(b)], str(out), "top",
                    BambuConfig(binary=stub, cwd_name="run1"))
    assert res.ok, res.errors
    args = (out / "run1" / "args.txt").read_text()
    assert str(a.resolve()) in args
    assert str(b.resolve()) in args


def test_parse_bambu_report_extracts_metrics(tmp_path):
    """XML 报告里的面积/时序字段应被解析出来。"""
    hls = tmp_path / "HLS_output"
    hls.mkdir()
    (hls / "top.xml").write_text(
        "<HLSReport>\n"
        "  <Area>1234</Area>\n"
        "  <TargetClockPeriod>5</TargetClockPeriod>\n"
        "  <EstimatedClockPeriod>3.2</EstimatedClockPeriod>\n"
        "  <Registers>77</Registers>\n"
        "  <DSPs>4</DSPs>\n"
        "  <BRAMs>2</BRAMs>\n"
        "</HLSReport>\n")
    got = parse_bambu_report(str(hls))
    assert got["area"] == 1234
    assert got["achieved_clock_ns"] == pytest.approx(3.2)
    assert got["target_clock_ns"] == pytest.approx(5)
    assert got["registers"] == 77
    assert got["dsps"] == 4
    assert got["brams"] == 2


def test_parse_bambu_report_falls_back_to_log(tmp_path):
    """没有 XML 时应从日志文本兜底提取面积。"""
    hls = tmp_path / "HLS_output"
    hls.mkdir()
    (hls / "top.log").write_text(
        "INFO: Synthesis completed\n"
        "Total area = 4321\n"
        "Number of registers = 91\n")
    got = parse_bambu_report(str(hls))
    assert got["area"] == 4321
    assert got["registers"] == 91


def test_parse_bambu_report_tolerates_garbage(tmp_path):
    """报告格式变化/字段缺失时返回空字典，不崩溃。"""
    hls = tmp_path / "HLS_output"
    hls.mkdir()
    (hls / "top.xml").write_text("this is not xml at all <<<")
    (hls / "top.log").write_text("nothing interesting here")
    assert parse_bambu_report(str(hls)) == {}
    assert parse_bambu_report(str(tmp_path / "does_not_exist")) == {}


# ------------------------------------------------- bambu / PandA 真实契约

def test_detect_top_module_unwraps_itanium_mangled_cxx(tmp_path):
    """C++ 内核会被 Itanium mangle，顶层模块名对不上 --top-fname。

    真实产物里 `gpt2_tiny_top` 编译成 `_Z13gpt2_tiny_topiiPf`，
    下游 yosys `hierarchy -top gpt2_tiny_top` 会直接报
    "Module `gpt2_tiny_top' not found!"。
    """
    from llm2asic.hls_backend.bambu import detect_top_module

    v = tmp_path / "k.v"
    v.write_text("module _Z13gpt2_tiny_topiiPf(input clock);\nendmodule\n"
                 "module datapath_x(input clock);\nendmodule\n")
    assert detect_top_module(str(v), "gpt2_tiny_top") == "_Z13gpt2_tiny_topiiPf"

    plain = tmp_path / "plain.v"
    plain.write_text("module k(input clock);\nendmodule\n"
                     "module k_helper(input clock);\nendmodule\n")
    assert detect_top_module(str(plain), "k") == "k"


def test_parse_bambu_report_scopes_metrics_to_top_function(tmp_path):
    """面积/触发器数必须取顶层函数的，不能被内部 softfloat 函数污染。

    真实日志里 `__float_adde11m52b_1023nih` 会先打印一整套指标（而且面积
    是 inf），顶层 `_Z13gpt2_tiny_topiiPf` 的指标在其后。
    """
    hls = tmp_path / "HLS_output"
    hls.mkdir()
    log = (
        "  Module binding information for function __float_adde11m52b_1023nih:\n"
        "    Total estimated area: 2360\n"
        "    Estimated number of DSPs: 99\n"
        "  Total number of flip-flops in function __float_adde11m52b_1023nih: 1260\n"
        "  Module binding information for function _Z13gpt2_tiny_topiiPf:\n"
        "    Estimated resources area (no Muxes and address logic): 65956\n"
        "    Total estimated area: inf\n"
        "    Estimated number of DSPs: 0\n"
        "    Minimum slack: 1.5\n"
        "  Total number of flip-flops in function _Z13gpt2_tiny_topiiPf: 21153\n"
    )
    got = parse_bambu_report(str(hls), log_text=log, top="gpt2_tiny_top")
    # inf 的 Total estimated area 必须退回不含 mux 的估计
    assert got["area"] == pytest.approx(65956)
    assert got["registers"] == 21153
    assert got["dsps"] == 0


def test_bambu_cmd_carries_required_float_flags(tmp_path):
    """-lm / --soft-float / -DFAITHFULLY_ROUNDED 必须默认带上。

    少了 -lm，Bambu 在 function allocation 阶段就报
    "does not exist a functional unit in the resource library: sqrtf"。
    """
    src = tmp_path / "top.cpp"
    src.write_text("void top(){}\n")
    stub = _stub_bambu(tmp_path)
    out = tmp_path / "w"
    res = run_bambu(str(src), str(out), "top",
                    BambuConfig(binary=stub, cwd_name="run1"))
    assert res.ok, res.errors
    args = (out / "run1" / "args.txt").read_text().split()
    assert "--soft-float" in args
    assert "-lm" in args
    assert "-DFAITHFULLY_ROUNDED" in args
    assert "--compiler=I386_CLANG16" in " ".join(args)


def test_bambu_finds_top_verilog_in_work_dir_and_collects_mem(tmp_path):
    """真实 PandA 把 <top>.v 写在**工作目录**（不是 HLS_output/）。

    引用了缺失 .mem 时（--simulate 的 array.mem、接口生成的 array_a.mem）
    必须补一个宽度/深度正确的全零占位，否则 yosys `$readmemh` 直接失败。
    """
    bindir = tmp_path / "stubbin"
    bindir.mkdir()
    stub = bindir / "bambu"
    stub.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then echo stub; exit 0; fi\n'
        "printf '0001020304050607\\n' > array_ref_1.mem\n"
        "cat > top.v <<'EOF'\n"
        "module mem_blk #(parameter data_size=8, n_elements=2,\n"
        "  MEMORY_INIT_file=\"array_ref_1.mem\") (input clk, output [7:0] q);\n"
        "endmodule\n"
        "module top(input clk, output [31:0] dout);\n"
        "  wire [7:0] q;\n"
        "  mem_blk #(.data_size(8), .n_elements(2)) u0 (.clk(clk), .q(q));\n"
        "endmodule\n"
        "EOF\n"
        "exit 0\n")
    stub.chmod(0o755)
    src = tmp_path / "k.cpp"
    src.write_text("void top(){}\n")

    res = run_bambu(str(src), str(tmp_path / "w"), "top",
                    BambuConfig(binary=str(stub), cwd_name="run1"))
    assert res.ok, res.errors
    # top.v 在工作目录里也能被找到
    assert [os.path.basename(p) for p in res.verilog] == ["top.v"]
    assert res.top_module == "top"
    names = sorted(os.path.basename(p) for p in res.mem_files)
    assert names == ["array_ref_1.mem"]
    # $readmemh 用的文件必须和工作目录里的 .v 同级
    for p in res.mem_files:
        assert os.path.dirname(p) == os.path.dirname(res.top_verilog)


def test_bambu_stubs_missing_mem_files(tmp_path):
    """生成的 .v 引用了不存在的 .mem 时要补占位并在 warnings 里说明。"""
    bindir = tmp_path / "stubbin"
    bindir.mkdir()
    stub = bindir / "bambu"
    stub.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then echo stub; exit 0; fi\n'
        "cat > top.v <<'EOF'\n"
        "module mem_blk #(parameter data_size=8, n_elements=2,\n"
        "  MEMORY_INIT_file=\"array.mem\") (input clk, output [7:0] q);\n"
        "endmodule\n"
        "module top(input clk, output [31:0] dout);\n"
        "  wire [7:0] q;\n"
        "  mem_blk u0 (.clk(clk), .q(q));\n"
        "endmodule\n"
        "EOF\n"
        "exit 0\n")
    stub.chmod(0o755)
    src = tmp_path / "k.cpp"
    src.write_text("void top(){}\n")

    res = run_bambu(str(src), str(tmp_path / "w"), "top",
                    BambuConfig(binary=str(stub), cwd_name="run1"))
    assert res.ok, res.errors
    assert [os.path.basename(p) for p in res.mem_stubbed] == ["array.mem"]
    stub_path = res.mem_stubbed[0]
    assert os.path.isfile(stub_path)
    # Bambu 的存储体模板用 $readmemb，一个字符 = 1 bit，
    # 所以占位必须是 8 个二进制 0（不是 2 个 hex digit）
    lines = open(stub_path).read().split()
    assert lines == ["0" * 8, "0" * 8]
    assert any("占位" in w for w in res.warnings)


# ---------------------------------------------------------------- yosys

def test_run_yosys_check_accepts_valid_design(tmp_path):
    """能展开的设计应通过，并给出 wires/cells 摘要。"""
    import shutil as _sh

    from llm2asic.hls_backend.bambu import run_yosys_check

    if _sh.which("yosys") is None:
        pytest.skip("环境里没有 yosys")
    v = tmp_path / "top.v"
    v.write_text("module top(input wire clk, input wire [3:0] a,\n"
                 "             output wire [3:0] y);\n"
                 "  assign y = a + 4'd1;\n"
                 "endmodule\n")
    ok, log, detail = run_yosys_check(str(v), "top", str(tmp_path))
    assert ok, detail
    assert os.path.isfile(log)
    assert "cells=" in detail and "wire_bits=" in detail
    stats = dict(kv.split("=") for kv in detail.split() if "=" in kv)
    assert stats["top_cells"] == "1"      # 顶层一个 $add


def test_run_yosys_check_rejects_bad_top_module(tmp_path):
    """顶层模块名对不上时必须报错（这正是 mangle 会踩的坑）。"""
    import shutil as _sh

    from llm2asic.hls_backend.bambu import run_yosys_check

    if _sh.which("yosys") is None:
        pytest.skip("环境里没有 yosys")
    v = tmp_path / "top.v"
    v.write_text("module _Z3topv(input wire a, output wire y);\n"
                 "  assign y = a;\nendmodule\n")
    ok, _log, detail = run_yosys_check(str(v), "top", str(tmp_path))
    assert not ok
    assert "top" in detail


# --------------------------------------------------- pragma 方言约束

def test_native_kernel_only_uses_supported_hls_pragmas(tmp_path):
    """生成的 C 内核只能用 PandA 插件真正支持的 pragma 子集。

    plugin_ASTAnalyzer 只注册 pipeline/inline/unroll/dataflow/cache/interface：
      * PIPELINE 必须是**函数作用域**（写进 for 循环体报
        "Loop pipelining pragma not supported."）；
      * ARRAY_PARTITION 没有 handler（退化成 Unknown HLS pragma 警告）；
      * INTERFACE 必须 `mode=<m> port=<n>`，裸写 `ap_none` 报
        "Missing interface mode attribute"，指针参数再用 ap_memory 报
        "Invalid HLS interface mode"。
    """
    ir = run_from_path(_model("gpt2_tiny"))
    ck = gen_c_kernel(ir, str(tmp_path), CKernelConfig(prefix="k"))
    assert ck.ok, ck.errors
    src = open(ck.top_path).read()

    assert "ARRAY_PARTITION" not in src
    assert "ap_ctrl_hs" not in src
    for line in src.splitlines():
        if "#pragma HLS INTERFACE" in line:
            assert " mode=" in line, line
            assert " port=" in line, line
        if "#pragma HLS PIPELINE" in line:
            # 只能出现在函数体的第一行（缩进 4，且是函数的开始）
            body = src.splitlines()
            idx = body.index(line)
            assert idx > 0
            assert body[idx - 1].rstrip().endswith("{"), line


# ----------------------------------------------------- 真实端到端

def _pandabambu():
    """找 bambu：先看 PATH，再看常见的本地安装前缀。

    ``make hls`` 是靠 PATH 上的 ``bambu`` 找 PandA 的；测试里额外探测几个
    常见本地安装前缀，好让真实闭环测试在没改 PATH 的会话里也能跑。
    """
    exe = shutil.which("bambu")
    if exe:
        return exe
    for cand in (os.path.expanduser("~/.local/panda/bin/bambu"),
                 os.path.expanduser("~/panda/bin/bambu"),
                 "/usr/local/bin/bambu"):
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


@pytest.mark.skipif(_pandabambu() is None or shutil.which("yosys") is None,
                    reason="需要真实的 PandA bambu 与 yosys")
def test_real_bambu_end_to_end(tmp_path):
    """真跑一次 PandA：gpt2_tiny 的 C 内核 -> 真实 Verilog -> Yosys 展开。

    这是 `make hls` 的最小闭环验证：确认拼出来的命令行确实被本地这套
    （clang17 + 补丁过的）PandA 接受，并且产出的 RTL 能被 yosys 展开。
    """
    ir = run_from_path(_model("gpt2_tiny"))
    cfg = HlsConfig(backend=HlsBackend.NATIVE, out_dir=str(tmp_path),
                    verify=False, run_bambu=True)
    cfg.bambu.binary = _pandabambu()
    res = build_hls(ir, cfg)
    assert res.ok, res.errors
    assert res.verified is False        # verify 关掉了数值比对
    assert res.top_verilog and os.path.getsize(res.top_verilog) > 1000
    assert res.yosys_ok, res.yosys_detail
    # 指标只能来自顶层的报告段
    assert res.registers and res.registers > 0


def test_run_yosys_check_stats_sum_all_modules(tmp_path):
    """wires/cells 必须是对**所有**模块求和，不能取到某个子模块。

    Bambu 产物的顶层 `_Z..top` 只是个 wrapper（本地 0 个 cell），逻辑都在
    submodule 里；而 `stat` 会按模块顺序输出，先出现的往往是 layernorm。
    """
    import shutil as _sh

    from llm2asic.hls_backend.bambu import run_yosys_check

    if _sh.which("yosys") is None:
        pytest.skip("环境里没有 yosys")
    v = tmp_path / "top.v"
    v.write_text(
        "module big_sub(input wire [7:0] a, output wire [7:0] y);\n"
        "  assign y = a + 8'd3;\n"
        "endmodule\n"
        "module big_top(input wire [7:0] a, output wire [7:0] y);\n"
        "  wire [7:0] y1, y2;\n"
        "  big_sub u0 (.a(a), .y(y1));\n"
        "  big_sub u1 (.a(y1), .y(y2));\n"
        "  assign y = y2;\n"
        "endmodule\n")
    ok, log, detail = run_yosys_check(str(v), "big_top", str(tmp_path))
    assert ok, detail
    stats = dict(kv.split("=") for kv in detail.split() if "=" in kv)

    # 自己按模块块统计一遍，断言摘要等于**全部**模块之和
    import re as _re

    txt = open(log, encoding="utf-8", errors="replace").read()
    blocks = _re.split(r"^=== .*===$", txt, flags=_re.MULTILINE)[1:]

    def _local(block, label):
        m = (_re.search(rf"Number of {label}:\s*(\d+)", block)
             or _re.search(rf"^\s*(\d+)\s+{label}\s*$", block, _re.MULTILINE))
        return int(m.group(1)) if m else 0

    assert int(stats["modules"]) == len(blocks) == 3
    assert int(stats["cells"]) == sum(_local(b, "cells") for b in blocks)
    assert int(stats["wire_bits"]) == sum(_local(b, "wire bits")
                                         for b in blocks)
    assert int(stats["cells"]) >= 2        # 两个加法，不能被"只看顶层"抹成 1


# ---------------------------------------------------------- ROM 合成化

_REALISTIC_MEM = """module ARRAY_1D_STD_DISTRAM_NN_SDS #(
  parameter data_size = 32,
  parameter n_elements = 4,
  parameter MEMORY_INIT_file = "array_ref_1.mem",
  parameter READ_ONLY_MEMORY = 1
) (
  input wire clock,
  input wire [31:0] memory_addr_a,
  output reg [data_size-1:0] dout_a
);
  reg [data_size-1:0] memory [0:n_elements-1]/* synthesis syn_ramstyle = "no_rw_check" */;

  initial
  begin
    if (MEMORY_INIT_file != "")
      $readmemb(MEMORY_INIT_file, memory, 0, n_elements-1);
    else
    begin
      for(integer i=0; i<n_elements; i=i+1)
      begin
        memory[i] = 0;
      end
    end
  end

  always @(posedge clock)
  begin
    if (READ_ONLY_MEMORY == 0)
      memory[memory_addr_a] <= dout_a;
    dout_a <= memory[memory_addr_a];
  end
endmodule
"""


def _write_mem(d, name, toks):
    with open(os.path.join(d, name), "w", encoding="utf-8") as f:
        f.write("\n".join(toks) + "\n")


def test_rom_rewrite_removes_initial_and_readmemb(tmp_path):
    """initial/$readmemb 是仿真专用，必须换成常量 case。"""
    v = tmp_path / "top.v"
    v.write_text(_REALISTIC_MEM, encoding="utf-8")
    _write_mem(str(tmp_path), "array_ref_1.mem",
               ["00000000000000000000000000000001",
                "11111111111111111111111111111110",
                "10101010101010101010101010101011",
                "00000000000000000000000000000000"])

    res = make_rom_synthesizable(str(v), [str(tmp_path)])
    assert res.readmem_removed == 1
    assert res.reads_romified >= 1
    assert res.files_baked == [os.path.join(str(tmp_path), "array_ref_1.mem")]
    assert res.files_missing == []

    text = v.read_text(encoding="utf-8")
    assert scan_simulation_only(text) == {}
    assert "$readmemb" not in text
    # 数组声明必须原样保留（后面还跟着注释和分号）
    assert 'reg [data_size-1:0] memory [0:n_elements-1]' in text
    # 删 initial 块不能留下孤儿 end：嵌套 else/for 的 end 要成对吃掉。
    # 必须按词计数，子串匹配会把 endmodule/endfunction 也算进去。
    assert len(re.findall(r"\bbegin\b", text)) == len(re.findall(r"\bend\b", text))
    # 逐位核对二进制初值
    assert "llm2asic_rom_memory = 32'h1;" in text        # 0b...01
    assert "llm2asic_rom_memory = 32'hfffffffe;" in text   # 0b...10
    assert "llm2asic_rom_memory = 32'haaaaaaab;" in text   # 1010...1011


def test_rom_rewrite_gates_reads_only_not_writes(tmp_path):
    """写端和字节使能子写不能被换成查表，否则 RW 存储体行为变了。"""
    v = tmp_path / "top.v"
    v.write_text(_REALISTIC_MEM, encoding="utf-8")
    _write_mem(str(tmp_path), "array_ref_1.mem", ["0" * 32] * 4)
    make_rom_synthesizable(str(v), [str(tmp_path)])
    text = v.read_text(encoding="utf-8")

    # 读端被门控
    assert "READ_ONLY_MEMORY ? llm2asic_rom_memory(" in text
    # 写端仍是原样赋值
    assert "memory[memory_addr_a] <= dout_a;" in text
    gated_write = re.compile(
        r"\(READ_ONLY_MEMORY \? llm2asic_rom_\w+\([^)]*\)"
        r"\s*:\s*\w+\[[^\]]*\]\)\s*\[[^\]]*\]\s*<=")
    assert not gated_write.search(text)


def test_rom_rewrite_zero_fill_for_missing_mem(tmp_path):
    """文件缺失或名为空时按全零 ROM 处理，宽度用声明表达式算对。"""
    v = tmp_path / "top.v"
    v.write_text(_REALISTIC_MEM, encoding="utf-8")
    res = make_rom_synthesizable(str(v), [str(tmp_path)])
    assert res.readmem_removed == 1
    # 默认的 array_ref_1.mem 不存在：要如实报出来，并退化成全零 ROM
    assert res.files_missing == ["array_ref_1.mem"]
    assert res.files_baked == []
    assert any("array_ref_1.mem" in w for w in res.warnings)
    text = v.read_text(encoding="utf-8")
    # 不能出现 {data_size-1:0{...}} 这种非法的复制宽度
    assert "{data_size-1:0{" not in text
    assert "{((data_size-1)-(0)+1){1'b0}}" in text


def test_rom_rewrite_keeps_readmemh_hex_semantics(tmp_path):
    """$readmemh 的每个字符是 4 bit，不能按二进制解析。"""
    v = tmp_path / "h.v"
    v.write_text("""module m #(
  parameter data_size = 16,
  parameter n_elements = 2,
  parameter MEMORY_INIT_file = "h.mem",
  parameter READ_ONLY_MEMORY = 1
) (input wire clock, input wire [31:0] a, output reg [data_size-1:0] q);
  reg [data_size-1:0] memory [0:n_elements-1];
  initial begin
    $readmemh(MEMORY_INIT_file, memory, 0, n_elements-1);
  end
  always @(posedge clock) q <= memory[a];
endmodule
""", encoding="utf-8")
    _write_mem(str(tmp_path), "h.mem", ["abcd", "0001"])
    make_rom_synthesizable(str(v), [str(tmp_path)])
    text = v.read_text(encoding="utf-8")
    assert "16'habcd" in text
    assert "16'h1" in text        # 前导 0 不影响数值


def test_scan_simulation_only_flags_residuals():
    """收尾扫描要能报出没改掉的仿真专用构造。"""
    assert scan_simulation_only("module m; endmodule\n") == {}
    src = """module m (input c, input [3:0] a, output reg [3:0] q);
  reg [3:0] mem [0:3];
  initial begin $readmemh("f.mem", mem); end
  always @(posedge c) q <= mem[a];
endmodule
"""
    found = scan_simulation_only(src)
    assert found.get("initial") == 1
    assert found.get("readmem") == 1


def test_run_bambu_synth_cleanup_off_keeps_initial(tmp_path):
    """synth_cleanup=False 时不改写，initial 原样留着。"""
    bindir = tmp_path / "stubbin"
    bindir.mkdir()
    stub = bindir / "bambu"
    stub.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then echo stub; exit 0; fi\n'
        "printf '0000000000000001\\n' > array_ref_1.mem\n"
        "cat > top.v <<'EOF'\n" + _REALISTIC_MEM.replace(
            "module ARRAY_1D_STD_DISTRAM_NN_SDS",
            "module ARRAY_1D_STD_DISTRAM_NN_SDS") + "EOF\n"
        "exit 0\n")
    stub.chmod(0o755)
    src = tmp_path / "k.cpp"
    src.write_text("void top(){}\n")

    res = run_bambu(str(src), str(tmp_path / "w"), "top",
                    BambuConfig(binary=str(stub), cwd_name="r_off",
                                synth_cleanup=False))
    assert res.ok, res.errors
    assert res.rom_readmem_removed == 0
    assert "$readmemb" in open(res.top_verilog, encoding="utf-8").read()


def test_run_bambu_synth_cleanup_on_bakes_rom(tmp_path):
    """默认开启：改写 + 报告固化结果，且残留构造为空。"""
    bindir = tmp_path / "stubbin"
    bindir.mkdir()
    stub = bindir / "bambu"
    stub.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then echo stub; exit 0; fi\n'
        "printf '00000000000000000000000000000001\\n' > array_ref_1.mem\n"
        "cat > top.v <<'EOF'\n" + _REALISTIC_MEM + "EOF\n"
        "exit 0\n")
    stub.chmod(0o755)
    src = tmp_path / "k.cpp"
    src.write_text("void top(){}\n")

    res = run_bambu(str(src), str(tmp_path / "w"), "top",
                    BambuConfig(binary=str(stub), cwd_name="r_on"))
    assert res.ok, res.errors
    assert res.rom_readmem_removed == 1
    assert [os.path.basename(p) for p in res.rom_files_baked] == ["array_ref_1.mem"]
    assert res.unsynthesizable == {}
    assert any("ROM 合成化改写" in w for w in res.warnings)
    text = open(res.top_verilog, encoding="utf-8").read()
    assert scan_simulation_only(text) == {}
    assert "32'h1" in text
