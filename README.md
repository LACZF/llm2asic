# LLM2ASIC

将大语言模型（LLM）权重编译为定制化 RTL 硬件电路的前端编译器。

## 快速开始

```bash
pip install -e .[dev]

# 端到端：把一份小模型配置 + 权重编译为 RTL，并跑仿真验证
llm2asic build --model examples/llama_tiny/model.yaml \
               --config examples/llama_tiny/config.yaml \
               --out ./build_out

# 运行测试
pytest
```

## 架构

Compiler pipeline: **Parser -> Quantizer -> Architect -> RTL Backend**

- `Parser`:   读取模型/权重，产出静态形状的 `LLM-IR`
- `Quantizer`: 量化权重，产出 `QLLM-IR` + 权重 ROM + 量化元数据
- `Architect`: 综合出参数化硬件架构描述 `ArchDesc`
- `RTL Backend`: 由 `ArchDesc` 生成可综合 Verilog + testbench + 综合脚本

`Parser` 之后还有一条**平行的 HLS 路径**，两者互不影响：

    GraphIR -> HLS C++(/ONNX) -> Bambu (PandA) -> Verilog

这条路径不做量化，直接在浮点 HLS 内核上编译，适合快速拿到可综合的 Verilog
并用商用/开源 HLS 工具做时序优化。详见 [HLS 后端](#hls-后端)。

## 输入模型格式

Parser 支持多种开箱可用的模型格式，统一归一化为 llama 系 canonical 权重：

| 路径 | 说明 |
|------|------|
| `examples/llama_tiny/model.yaml` | 声明式 `.yaml` + `.npz`（纯 NumPy，无需任何第三方） |
| `examples/llama_mini/model.yaml` | 同上，微型 demo（含 ASIC 综合配置） |
| `examples/llama_tiny_safetensors/` | `.safetensors` + `config.json`（HF 权重） |
| `examples/llama_tiny_onnx/` | `model.onnx` + `config.json`（ONNX 权重） |
| `examples/llama_tiny_bin/` | 原始 fp32 `.bin` + `model.yaml`（exporter 产物） |
| 任意 `torch.nn.Module` / HF `.bin` | 需安装 torch（可选路径） |

```bash
# 用不同格式跑同一模型（结果逐位一致）
llm2asic build --model examples/llama_tiny_safetensors/model.safetensors ...
llm2asic build --model examples/llama_tiny_onnx/model.onnx ...
llm2asic build --model examples/llama_tiny_bin/model.bin ...
```

## HLS 后端

`llm2asic hls` 把 `GraphIR`（parser 产物）编译成 HLS C++，可选再交给
[Bambu (PandA HLS)](https://github.com/ferrandi/PandA-bambu) 综合成 Verilog。
这与 `llm2asic build` 的原生 RTL 流程完全独立 —— 不做量化，走浮点 HLS。

```bash
pip install -e .[hls]          # onnx + hls4ml

# native C 内核（默认，无需第三方 HLS 工具）
llm2asic hls --model examples/llama_tiny/model.yaml --out ./build_hls

# 只产出 C++，不调用 Bambu（没有 HLS 工具链时用）
llm2asic hls --model examples/llama_tiny/model.yaml --out ./build_hls --no-bambu

# 只要 Verilog，跳过慢的 Yosys 展开检查
llm2asic hls ... --no-yosys-check

# 经 hls4ml 生成 HLS C++ 工程；或只导出 ONNX
llm2asic hls ... --hls-backend hls4ml
llm2asic hls ... --hls-backend onnx
```

Makefile 快捷目标：

```bash
make hls                                  # native C 内核
make hls       HLS_BACKEND=hls4ml
make hls-all                               # 三条路径各跑一遍
```

`make hls` 会**自动探测 `bambu`**：装了 Bambu 就一路走到 Verilog；没装就停在
C++/ONNX 并打印提示（产物依然可用，且 native 路径带数值自检）。想强制要求
Bambu 时用 `make hls HLS_STRICT=1`，此时缺 Bambu 会直接失败。

### 三条路径

| `--hls-backend` | 产物 | 说明 |
|---|---|---|
| `native` | 自包含 C++ 内核 + Bambu Verilog | 自研生成器，支持 float/double；`emit_main` 可跑 g++ 数值自检 |
| `hls4ml` | hls4ml HLS C++ 工程 | 走 `ModelGraph.from_layer_list`；用 hls4ml 自己的 nnet 算子库 |
| `onnx` | 单步 decode ONNX | 便于接入其它 HLS/仿真工具链 |

### native + Bambu 闭环（已实测）

`native` 路径默认会把生成的 C++ 交给 Bambu，再用 Yosys 检查 RTL：

```
[llm2asic] warn: 为 2 个缺失的初始化文件生成了全零占位: array.mem, array_a.mem
[llm2asic] warn: yosys 展开通过 modules=1241 cells=29110 wire_bits=1627593 memories=83 memory_bits=152064 top_cells=0 check_warnings=1
[llm2asic] PASS: backend=native ok=True c_kernel=gpt2_tiny_kernel.cpp verilog=gpt2_tiny_top.v area=65956 ff=21153 yosys=True verified=True
```

产物布局（`OUT` 为输出根目录）：

| 文件 | 说明 |
|---|---|
| `hls/gpt2_tiny_kernel.cpp` | 生成的内核（可单独 g++ 编译） |
| `hls/bambu/bambu_run/gpt2_tiny_top.v` | Bambu 综合出的 RTL |
| `hls/bambu/bambu_run/*.mem` | ROM 初始化数据；缺失的按 `data_size`/`n_elements` 补全零占位 |
| `hls/bambu/bambu_run/bambu.log` | Bambu 完整 stdout |
| `hls/bambu/bambu_run/yosys_check.log` | Yosys 展开日志 + `stat` |
| `hls/bambu/bambu_run/yosys_check.ys` | 实际执行的 Yosys 脚本 |

实现上踩过的坑（已在代码里处理，改配置时注意）：

- **顶层模块名会被 mangle**。C++ 内核综合出来的顶层是
  `_Z13gpt2_tiny_topiiPf` 而不是 `gpt2_tiny_top`，所以下游 Yosys 用的是探测出来的
  真实模块名，不是 `--top-fname`。
- **PandA 的 pragma 子集很窄**。只发它真正实现了的几种：`PIPELINE` 必须写在
  函数作用域（循环体内会报 `Loop pipelining pragma not supported.`）；
  `ARRAY_PARTITION` 没有插件 handler；`INTERFACE` 必须带 `mode=<m> port=<n>`，
  且 `float *` 不能用 `ap_memory`；`ap_ctrl_hs` 不支持。
  `pipeline_ii` 默认 **0 = 不发** `PIPELINE`。
- **libm 要 soft-float + `-lm`**。默认带 `--soft-float`、`-lm`、
  `-DFAITHFULLY_ROUNDED`；否则会在 function allocation 阶段报
  `does not exist a functional unit in the resource library: sqrtf`。
  另外本仓库的 PandA 前端需要 `-fno-builtin`（已内置）。
- **面积指标可能打成 `inf`**。总面积含 mux/地址逻辑时会溢出，报告里改用
  `Estimated resources area (no Muxes and address logic)` 的有限值兜底，
  并且只在**顶层函数**的报告段里取值（内部 softfloat helper 的指标在前面）。
- Yosys 用 `check` 而不是 `check -assert`：Bambu 顶层会留一个没人驱动的
  `OUT_UNBOUNDED_*` 通道，`-assert` 会让整条流程失败；告警数会记进摘要。

`gpt2_tiny` 实测：顶层 1241 个模块 / 29110 个 cell / 83 个 memory
（152064 bit），Bambu 报 `area=65956`、`ff=21153`。gpt2_tiny 的 Yosys 检查
约需 10 分钟（1241 个模块），只想要 Verilog 时用 `--no-yosys-check`。

### 数值校验

`native` 路径默认（`verify: true`）会用 `g++` 编译生成的 C 内核，并与 numpy
浮点参考 `ref_decode_step()` 在多组 `(token, pos)` 上比对，相对误差容差
`rel_tol`（默认 `1e-4`）。这不依赖任何 HLS 工具链，是当前最强的正确性保证：

```
[llm2asic] PASS: backend=native ok=True c_kernel=llm2asic_kernel.cpp verified=True
```

`hls4ml` 路径会在生成后用 `g++ -fsyntax-only` 检查可编译性；其数值语义与
native 路径**未**做等价性验证，仅保证工程能编译。

### 当前限制

- 单步 decode 语义：KV 长度按 1 处理，attention 退化为 `v` 透传，
  不覆盖自回归多步推理（`pos` 是编译期常量）。
- Bambu 未安装时 `llm2asic hls` 会明确报错并返回非 0，不会静默跳过。
  注意 PyPI 上的 `bambu` 包与 PandA 无关。
- hls4ml 生成的 C++ 使用 Vivado 风格的 `ap_*`，与 Bambu 的兼容性尚未实测。
- Bambu 合成不出初始化 `.mem` 时会按零填充占位（数值上等价于全零权重，
  但不是模型真值）；真实权重要自己放 `.mem`。
- 当前设计规模很大（`gpt2_tiny` 顶层 29110 个 cell），未经 memory
  partitioning / 跨迭代调度，时序与面积都不是最优。
- Yosys 只做展开检查（`read_verilog`/`hierarchy`/`proc`/`opt`/`check`/`stat`），
  **不做**工艺映射，也不产生网表。

### 配置

在模型配置 YAML 里加 `hls:` 段（CLI 参数优先级更高）：

```yaml
hls:
  backend: native        # native | hls4ml | onnx
  pos: 0                 # 编译期已知的位置索引
  run_bambu: true
  verify: true           # native: g++ 数值自检
  precision: float       # float | double
  n_buffers: 8           # C 内核每种宽度的缓冲池上限
  pipeline_ii: 0         # >0 -> `#pragma HLS PIPELINE II=N`；0=不发
  yosys_check: true      # Bambu 产物做 Yosys 展开检查
  yosys: yosys
  hls4ml_precision: float
  hls4ml_reuse_factor: 1
  hls4ml_io_type: io_parallel
  bambu:
    device_name: xc7a100t-1csg324-VVD
    clock_period: 5.0
    compiler: I386_CLANG16
    soft_float: true          # 关掉会缺 functional unit（sqrtf/expf…）
    faithful_rounding: true   # -DFAITHFULLY_ROUNDED
    link_libm: true           # -lm
    experimental_setup: ""    # 注意：自带 -O0，会覆盖 opt_level
    mem_stub: true            # 补齐缺失的 .mem（全零占位）
    synth_cleanup: true       # initial/$readmemb -> 常量 case ROM
    evaluation: ""       # 如 PERIOD,AREA,REGISTERS,DSPS,BRAMS
    simulate: false
```

缓冲池打满且没有可复用的空闲缓冲时会**直接报错**（而不是复用仍活跃的缓冲
产出错误结果），此时调大 `n_buffers` 即可。

### 可综合性：`initial` / `$readmemb`

Bambu 生成的存储体模板（`ARRAY_1D_STD_DISTRAM_NN_SDS`、
`STD_SP_BRAM`、`BRAM_MEMORY_CORE_SMALL` 等）都带这么一段：

```verilog
reg [data_size-1:0] memory [0:n_elements-1];
initial
begin
  if (MEMORY_INIT_file != "")
    $readmemb(MEMORY_INIT_file, memory, 0, n_elements-1);
  else
  begin
    for(index=0; index<n_elements; index=index+1)
    begin
      memory[index] = 0;
    end
  end
end
```

`initial` + `$readmemb` **只在仿真下有意义**：综合器不读外部 `.mem` 文件，
ASIC 流程会直接报错，FPGA 流程则常常静默丢初值——于是权重全零，`check`
照样过、`yosys` 照样过，只有对比仿真结果才发现数字不对。

`synth_cleanup: true`（默认开）在 Bambu 出 RTL 之后、收 `.mem` 之前，把它
就地改写成可综合的常量查表：

```verilog
function [data_size-1:0] llm2asic_rom_memory;
  input [31:0] llm2asic_rom_addr;
  begin
    llm2asic_rom_memory = {((data_size-1)-(0)+1){1'b0}};   // 默认全零
    if (MEMORY_INIT_file == "array_ref_30905.mem") begin
      case (llm2asic_rom_addr)
        0: llm2asic_rom_memory = 32'h1ffff;
        1: llm2asic_rom_memory = 32'h1fe02;
        ...
      endcase
    end
  end
endfunction
```

读端按只读标志分流，**写端和字节使能子写原样不动**：

```verilog
dout_a <= (READ_ONLY_MEMORY ? llm2asic_rom_memory(addr) : memory[addr]);
```

几点容易踩的地方，这里都处理了：

- **`$readmemb` 一个字符 = 1 bit**，不是十六进制。`array_ref_30905.mem`
  每行 32 个字符就是 32 bit，按 hex 解会得到 128 bit，数据全错。
- 宽度从声明表达式推（`data_size-1:0`、`BITSIZE_data_out-1:0`、
  `(n_byte_on_databus)*8-1:0`），不能当成常数写死。
- 删 `initial` 必须成对匹配 `begin`/`end`。Bambu 的块里有嵌套
  `else` + `for`，用非贪婪 `.*?end` 会停在 `for` 体的 `end`，留下一堆
  孤儿 `end`，Yosys 直接语法错。
- 数组声明末尾还跟着注释和分号（`reg [...] memory [0:n-1] /* ... */;`），
  ROM 函数必须插在**声明之前**，插在声明和分号之间会截断声明。
- 字节使能写是 `memory[a][i*8+:8] <= ...`，判断读/写要跳过尾随的位选，
  否则会把写操作门控成查表，读写存储体行为就变了。
- 找不到 `.mem` 时按全零 ROM 处理，并在 warnings 里点名——静默烤零正是
  最难查的那种错。

改写只动上面这些位置，其余字节不变。gpt2_tiny 实测：6 个存储体模板、
8 个 `.mem`，Yosys `check` 干净（只剩 Bambu 本来就有的那条
`OUT_UNBOUNDED_*` 无驱动告警），area/ff 与改写前一致（65956 / 21153），
1312 个地址的查表结果与 `$readmemb` 逐位相同。
