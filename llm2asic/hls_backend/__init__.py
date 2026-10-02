# llm2asic/hls_backend
"""LLM-IR -> HLS -> RTL 后端。

三条路径：

- `native`  : GraphIR -> 自研 HLS C 内核 -> Bambu -> Verilog
- `hls4ml`  : GraphIR -> hls4ml HLS C++（可选再交给 Bambu）
- `onnx`    : GraphIR -> ONNX

全部入口都是纯函数式：缺工具时返回带 `errors` 的结果对象，不抛异常。
"""

from .bambu import (
    BambuConfig,
    BambuResult,
    bambu_available,
    bambu_version,
    run_bambu,
)
from .c_kernel import CKernelConfig, CKernelError, CKernelResult, gen_c_kernel
from .float_ref import ref_decode_step, ref_decode_trace
from .hls4ml_gen import Hls4mlConfig, Hls4mlError, Hls4mlResult, build_hls4ml
from .pipeline import (
    HlsBackend,
    HlsConfig,
    HlsResult,
    build_hls,
    export_onnx,
)

__all__ = [
    "BambuConfig", "BambuResult", "bambu_available", "bambu_version",
    "run_bambu",
    "CKernelConfig", "CKernelError", "CKernelResult", "gen_c_kernel",
    "ref_decode_step", "ref_decode_trace",
    "Hls4mlConfig", "Hls4mlError", "Hls4mlResult", "build_hls4ml",
    "HlsBackend", "HlsConfig", "HlsResult", "build_hls", "export_onnx",
]
