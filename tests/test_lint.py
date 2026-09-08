# tests/test_lint.py
"""Verilator lint 脚本测试 (scripts/lint_rtl.sh):
干净设计 → PASS 报告; 出现未豁免告警类别(如 LATCH) → FAIL。"""

import pathlib
import shutil
import subprocess

import pytest

VERILATOR = shutil.which("verilator")
pytestmark = pytest.mark.skipif(VERILATOR is None, reason="需要 Verilator")

SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "lint_rtl.sh"

_CLEAN = {
    "sub_a.sv": (
        "module sub_a #(parameter W=8)(input clk, input [W-1:0] d,"
        " output reg [W-1:0] q);\n"
        "  always_ff @(posedge clk) q <= d;\n"
        "endmodule\n"
    ),
    "sub_b.sv": (
        "module sub_b #(parameter S=1)(input a, output b);\n"
        "  assign b = ~a;\n"
        "endmodule\n"
    ),
    "tiny_accel.sv": (
        "module tiny_accel(input clk, input [7:0] d, output [7:0] q, output b);\n"
        "  wire x;\n"
        "  sub_a #(.W(8)) ua(.clk(clk), .d(d), .q(q));\n"
        "  sub_b #(.S(1)) ub(.a(1'b0), .b(x));\n"
        "  assign b = x;\n"
        "endmodule\n"
    ),
}

_LATCH = {
    "tiny_accel.sv": (
        "module tiny_accel(input sel, input d, output reg q);\n"
        "  always @* begin if (sel) q = d; end\n"
        "endmodule\n"
    ),
}


def _run(tmp_path, files):
    d = tmp_path / "rtl"
    d.mkdir()
    for name, text in files.items():
        (d / name).write_text(text)
    report = tmp_path / "lint_report.txt"
    p = subprocess.run(
        ["bash", str(SCRIPT), "--rtl", str(d), "--report", str(report)],
        capture_output=True, text=True, timeout=300,
    )
    return p, report


def test_lint_clean_passes(tmp_path):
    p, report = _run(tmp_path, _CLEAN)
    assert p.returncode == 0
    assert report.exists()
    text = report.read_text()
    assert "verdict : PASS" in text
    assert "top 模块" in text or "顶层模块" in text


def test_lint_tracks_instantiated_submodules(tmp_path):
    """文件清单只含顶层 + 顶层 #() 实例化的模块(模块名=文件名)。"""
    p, report = _run(tmp_path, _CLEAN)
    assert p.returncode == 0
    text = report.read_text()
    assert "sub_a.sv" in text
    assert "sub_b.sv" in text
    assert "sim_tb.sv" not in text


def test_lint_fails_on_unexpected_warning(tmp_path):
    """非豁免类别(LATCH)必须使 lint FAIL 且退出码非 0。"""
    p, report = _run(tmp_path, _LATCH)
    assert p.returncode != 0
    assert report.exists()
    text = report.read_text()
    assert "verdict : FAIL" in text
    assert "LATCH" in text