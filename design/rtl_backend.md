# RTL 后端（RTL Backend）— RTL 生成、验证与部署

> 组件职责：**写出电路**。把 Architect 的 `ArchDesc`（架构蓝图）转化为**可综合的 Verilog/
> SystemVerilog**，生成权重 ROM 引用、顶层数据流网络、双模式控制器、测试平台与激励，并完成
> RTL 仿真数值比对与综合脚本产出。对应 `llm2asic/rtl_backend/`。

---

## 1. 目标与输入输出

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  输入:  ArchDesc (archdesc.json)                                              │
│         + weights_rom/* (来自 Quantizer，复制到输出)                            │
│         + quant_metadata.json (拆包/scale 定点信息)                            │
│         + 黄金数据 (examples 里导出的 fp16 参考输出)                            │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Step 1  引擎 RTL 生成   (每个 engine_type x 模板)                              │
│  Step 2  数据流与顶层生成  (层间 FIFO + 双模式控制器 + 接口包装)                 │
│  Step 3  测试平台与激励    (testbench + 384KV 输入 + 参考输出比对)              │
│  Step 4  仿真验证          (Verilator，三层一致性)                             │
│  Step 5  综合/部署脚本     (Vivado/Yosys + bitstream/GDS 驱动)                 │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  输出:  RTL 源码 + 权重 ROM 副本 + testbench + 综合脚本 + 资源/性能报告          │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 设计原则

1. **模板驱动**：每种 `engine_type` 一个 Verilog/SV 模板；`ArchDesc` 的字段通过模板引擎
   参数化填充（本项目用 Jinja2）。模板与 ArchDesc 引擎类型**一一对应**（`architect.md §7` 约定）。
2. **参数可综合**：`pe/simd/位宽/深度` 全部作为 Verilog `parameter` 传出，模块由参数实例化，
   保证可综合、可复用。
3. **AXI-Stream 握手**：引擎间用 `valid/ready` 流式协议 + FIFO 解耦（对应 `DataflowLink`）。
4. **确定性**：无随机、固定 seed、稳定文件布局，保证 build 可复现。
5. **旁路可观测**：每级输出落盘，黄金比对支持逐层定位。

---

## 3. Step 1：引擎 RTL 生成

### 3.1 模板目录
```
rtl_backend/templates/
├── gemm_engine.sv.tpl          # GEMV/GEMM 引擎
├── attention_engine.sv.tpl     # Attention（QK·V + mask + 在线 softmax 封装）
├── softmax_engine.sv.tpl       # 流式在线 softmax
├── rmsnorm_engine.sv.tpl       # RMSNorm/LayerNorm
├── embedding_engine.sv.tpl     # token→向量查表
├── activation_engine.sv.tpl    # silu/gelu/relu
├── kv_memory.sv.tpl            # KV 缓存 + 分区读写
├── fifo.sv.tpl                 # 同步/异步 FIFO
└── top.sv.tpl                  # 顶层数据流 + 控制器
```

### 3.2 GEMV 引擎模板要点（decode 核心）

沿用 `architect.md §4.1` 的"权重静止"结构：

```systemverilog
module gemm_engine #(
    parameter int PE = 16,          // 输出并行
    parameter int SIMD = 8,         // 输入并行
    parameter int INW = 8,          // 激活位宽
    parameter int WW = 4,           // 权重位宽
    parameter int ACCW = 32,        // 累加位宽
    parameter int OUTW = 8,
    parameter int ROM_DEPTH = 65536,
    parameter int MODE = 0          // 0=decode(gemv) 1=prefill(gemm)
)(
    input  logic clk, rst,
    // 激活输入 (AXI-Stream)
    input  logic [SIMD*INW-1:0] s_tdata,
    input  logic s_tvalid, output logic s_tready,
    // 输出
    output logic [PE*OUTW-1:0] m_tdata,
    output logic m_tvalid, input logic m_tready,
    // 权重(常量内嵌 ROM，经 $readmemh 初始化)
    // 简化示意：实际权重由地址生成器逐个读出
);
    logic [ROM_DEPTH-1:0] weight_mem [0:ROM_DEPTH-1];
    initial $readmemh("weights/gemm_0_weight.mem", weight_mem);

    logic [ACCW-1:0] acc [PE];
    generate
        for (genvar p = 0; p < PE; p++) begin : pe_arr
            // SIMD 个乘法 + 加法树 + 累加器
            // scale 定点化: acc = (acc * m) >> e  (来自 quant_metadata)
        end
    endgenerate
    // 地址生成器按 quant_metadata.layout 顺序出权重到各 PE
endmodule
```

> **权重 ROM 两种实现**（对应 `quantizer.md §4` 的权重布局与固化策略）：
> - **internal 内嵌**：`initial $readmemh` 或 RTL `parameter` 常量数组 —— 本项目主力，零加载延迟。
> - **external 外部**：运行时经 AXI 输入权重（当权重超片上容量时）。
> 顶层默认 internal；开门见山，`quant_metadata` 给出具体实现选择。

### 3.3 Attention 引擎（复用 GEMV）
`attention_engine` 内部例化 `gemm_engine`（QK 点积、AV 加权）+ `softmax_engine` + causal mask。
decode 为单 query，prefill 为 bmm，由 MODE 参数切换。

### 3.4 流式 Softmax（在线，`architect.md §4.6`）
单遍维护 `(m, sum)` 并在 AV 累加中重缩放，避免二次访问 K/V，节约带宽与时间。

---

## 4. Step 2：数据流与顶层生成

### 4.1 FIFO 插入
按 `aux flow`（`DataflowLink`）在引擎间插入 `fifo.sv`，深度取 ArchDesc 指定值，用于解耦
上下游吞吐与流水停顿。

### 4.2 顶层控制器（prefill/decode FSM）
顶层模块含一个 **mode 控制器**（`top.sv.tpl`）：
- 输入端口 `run_mode`（prefill/decode）
- 状态机依次驱动各引擎使能，管理 KV 缓存的写/读时序
- 在 decode 时按 token 循环：`读 KV→attention→gated_mlp→残差→输出 logits`

### 4.3 接口包装
顶层暴露统一 AXI-Stream 风格接口（数据 + valid/ready + tlast 表示序列/帧结束），
与 `ArchDesc.users_facing_interfaces` 对齐，便于集成到 SoC / 测试平台。

---

## 5. Step 3 & 4：测试平台与仿真验证

### 5.1 三层一致性验证（贯穿 Parser → Quantizer → Architect → RTL Backend）
| 层 | 输入 | 工具 | 用途 |
|----|------|------|------|
| 参考软件 | LLM-IR(f32/f16) | torch/numpy | 真值 |
| 量化模拟 | QLLM-IR(int) | torch/numpy 定点模拟 | 检验量化误差 |
| RTL | QLLM-IR 对应 RTL | **Verilator** | 检验实现正确性 |

每层之间做数值比对；`quant_metadata` + `ArchDesc` 生成测试向量。

### 5.2 Testbench 生成（`rtl_backend/testbench.py`）
- 从黄金数据读取一组 token 输入 + 期望 logits
- 打包为顶层接口格式（量化 + SIMD 打包，与 `quant_metadata.layout` 一致）
- 驱动 `run_mode` 依次跑 prefill 片段与 decode 逐步生成
- 比对输出 logits 的 top-k 命中与逐元素误差，输出 PASS/FAIL 报告

```python
def gen_testbench(archdesc, gold_data, out_dir):
    # 生成 top_tb.sv + 输入激励文件(inputs.mem) + 期望输出(expected.txt)
    ...
    run_verilator(out_dir)          # 编 RTL + tb，跑仿真
    report = compare(outputs, gold_data, tol=int8_tol)
    return report                   # → test_report.json
```

### 5.3 误差判据
- 分类指标：logits 的 top-1 / top-5 命中率与参考一致
- 数值指标：逐元素相对误差 / 余弦相似度，阈值由 `quant_cfg` 指定
- 失败时**定位到引擎**（per-engine trace），回填到编译报告

---

## 6. Step 5：综合与部署脚本

### 6.1 Vivado 流程（商业）
`rtl_backend/verify.py` 与 `deploy.tcl` 生成可复用的综合流：
```tcl
create_project -force llm_accel
add_files -norecurse [glob rtl/**/*.sv rtl/**/*.v]
add_files -fileset constrs_1 constraints.xdc
add_files weights/                          ;# ROM 文件路径供 $readmemh 引用
set_property top llm_accel [current_fileset]
launch_runs synth_1 -jobs 8; wait_on_run synth_1
report_utilization -file util_report.txt
# (可选) 布局布线 + write_bitstream
```
产出 `util_report.txt` 的资源使用，与 Architect 估算对比，用于 DSE 模型校准。

### 6.2 Yosys 流程（开源）
生成 `synth.ys`，支持两套后端（PDK/器件可配置）：

- **FPGA**（默认）：`read_verilog` + `synth_xilinx -family xc7 -flatten -nowidelut`，
  产出 `netlist.v` 与 `util_report.txt`。乘法/权重 ROM 由 Xilinx DSP48 / RAMB 硬块吸收，
  是本设计在开源工具下可完整跑通的综合路径（~20 min / ~9 GB 峰值）。
- **ASIC 标准单元**（`synth.backend = asic`）：`read_liberty -lib <lib>` +
  `memory -nomap` + `abc -liberty` + `dfflibmap`，映射到任意 PDK 的 `.lib`
  （默认自动探测 OpenROAD-flow-scripts 下的 `sky130hd/sky130hs/nangate45/asap7/gf180` 等）。

配置通过 `synth` 节（`backend`/`family`/`pdk`/`liberty`）或命令行
`--backend/--pdk/--liberty` 指定；Makefile 暴露 `BACKEND`/`PDK`/`LIBERTY` 变量，例如：

```sh
make synth BACKEND=asic PDK=sky130hd LIBERTY=/path/.../sky130_fd_sc_hd__tt_025C_1v80.lib
```

#### ASIC 综合的容量约束（重要）
受 11 GB 内存与开源 `abc` 能力的限制，`llama_tiny_accel`（乃至更小模型）的
**ROM 供数宽乘加（gemv 24×24、attn 64×64）数据通路**无法在合理时间内用
`abc -liberty` 映射成标准单元——该瓶颈与模型规模及 rsqrt 表大小无关（实测单个
`gemv_0` 模块的 `abc` 映射即长时间停滞），因此完整设计的 sky130 门级网表
当前无法在受限环境产出。FPGA 路径因硬块吸收这些资源而可行。
`memory -nomap` 避免把内嵌 ROM 展开（否则单个 2^20 项 rsqrt 表 ≈ 100 万单元 → OOM）。

### 6.3 rsqrt 查找表尺寸旋钮
`gen_luts` 的 rsqrt 表默认 `2^20` 项（`QuantConfig.rsqrt_lut_bits = 20`，与历史一致，
精度最高）。ASIC 演示可在配置 `quant.rsqrt_lut_bits` 中调小（如 10-12）以减小内存，
RTL 与黄金参考始终共用同一张表、保持逐位一致，代价是 rmsnorm 精度变粗。

### 6.4 ASIC（GDS）接入（远期）
跨过 FPGA/标准单元综合，将 RTL 交给 ASIC 后端（如开源 `OpenLane` / `OpenROAD`），
此路径在路线图 Phase 4（`top.md §9`）。

---

## 7. 输出规范

```
out/rtl/
├── rtl/                        # 可综合 RTL
│   ├── top.sv
│   ├── gemm_engine.sv  attention_engine.sv  softmax_engine.sv
│   ├── rmsnorm_engine.sv  embedding_engine.sv  activation_engine.sv
│   ├── kv_memory.sv  fifo.sv
│   └── ...
├── weights/                    # Quantizer ROM 副本
│   └── *.mem / *.coe
├── tb/                         # 测试平台
│   ├── top_tb.sv
│   ├── inputs.mem
│   └── expected.txt
├── scripts/
│   ├── vivado_build.tcl
│   ├── synth.ys
│   └── run_sim.sh              # 封装 Verilator 仿真
└── reports/
    ├── test_report.json        # 一致性验证结果
    ├── util_report.txt         # 资源占用
    └── perf_report.json        # 延迟/吞吐测量
```

---

## 8. 关键设计决策总结

| 决策点 | 推荐 | 依据 |
|--------|------|------|
| 生成语言 | Verilog / SystemVerilog | 可综合、工具通用 |
| 生成方式 | 模板驱动（Jinja2） | 可扩展、可参数化 |
| 权重 ROM | internal 内嵌（$readmemh） | 消除加载延迟（LLM decode 核心） |
| 层间协议 | AXI-Stream 风格 + FIFO | 标准、可综合 |
| 双模式 | 顶层 FSM 切换 prefill/decode | 复用引擎、降面积 |
| 验证 | 三层比对（参考/量化/RTL） | 可定位、可回归 |
| 综合 | Vivado（商）+ Yosys（开源） | 覆盖两种生态 |

---

## 9. 与上下游组件的契约回顾
- **输入**完全来自 `ArchDesc + quant_metadata + weights_rom`：RTL Backend **不重新解析模型**。
- **引擎类型集**与 Architect 模板一一对应：新增引擎需同时更新 `architect.md§7` 的 ArchDesc 类型
  与 `rtl_backend/templates/`。
- **权重布局**严格遵守 `quant_metadata.layout`（Quantizer 固定），RTL 的拆包逻辑不得漂移。

### 总结
RTL Backend 把 `ArchDesc` "实例化"为可综合 RTL：引擎模板化复用、FIFO 解耦、顶层双模式控制器、
$readmemh 固化权重、三层一致性验证、以及 Vivado/Yosys 双综合脚本，构成 LLM→RTL 流程的
**最终落地环节**，交付可直接综合/部署的硬件描述、测试平台与报告。
