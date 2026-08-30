# LLM2ASIC：将大语言模型权重编译为硬件电路 — 总体设计文档

> 本文档是 `llm2asic` 编译器的**顶层总纲**（Top Design）。它定义了整个编译器的使命、
> 核心设计理念、四层组件边界（Parser / Quantizer / Architect / RTL Backend）、贯穿全流程的中间表示（IR）契约、数据流、技术栈与路线图。
> 各组件的详细设计分别见 `parser.md` ~ `rtl_backend.md`。

---

## 1. 项目使命与设计目标

`llm2asic` 是一个**前端编译器（Ahead-of-Time Compiler）**，它将训练好的**大语言模型（LLM）**
的权重文件，一次性转化为一套针对该模型**定制化**的、可直接综合（synthesizable）的
**RTL（寄存器传输级）硬件描述**，进而可落地为 FPGA 比特流或 ASIC 版图（GDS）。

设计灵感源自 AMD 收购的 **Taalas** 以及 "weight-pixelization" 理念 —— **把权重直接"刻"
进硬件的晶体管/查找表中**，从而在推理时消除"加载权重"这一额外开销与带宽瓶颈，获得极致
的能效与延迟。

### 1.1 本编译器与通用 AI 加速器的本质区别

| 维度 | 通用 GPU / 数据中心加速器 | **LLM2ASIC（定制化编译器）** |
|------|--------------------------|------------------------------|
| 权重 | 运行时从 DRAM 加载 | **编译期固化**为 ROM / 逻辑常量，运行时零加载开销 |
| 算力映射 | 单一通用 SIMT/SIMD 阵列 | **逐层定制**的数据流引擎（每层一个专用引擎或参数化引擎） |
| 数据流 | 每次推理都重新搬运 | 层间流式直连，数据"留在片上" |
| 灵活性 | 任意模型 | **只针对被编译的那个模型**（换模型需重新编译） |
| 精度 | 通用 fp32/fp16 | 编译期决定位宽（INT8/INT4/INT2/...），可逐层/逐分组定制 |
| 目标 | 软件生态 | 专用推理 ASIC / FPGA |

> **关键取舍**：LLM2ASIC 用"灵活性"换取"极致能效"。它生成的是**面向单一模型**的专用电路，
> 因此要求模型结构在编译期**完全已知且确定**（静态形状、静态图）。

### 1.2 设计目标（非功能需求）

1. **正确性**：生成硬件推理结果与软件参考（fp32/bf16 仿真）数值一致（在容差范围内）。
2. **可综合**：输出为标准 Verilog / SystemVerilog，可通过 Vivado / 开源综合工具（如 Yosys）综合。
3. **可复现**：同输入权重 + 同配置 ⇒ 完全相同的输出文件（build 确定性）。
4. **可配置**：通过命令行/配置文件控制量化位宽、并行度、面积-吞吐权衡（DSE）。
5. **可扩展**：新增层类型/算子只需实现一个"模板 + 代码生成片段"，不改动框架骨架。
6. **可观测**：每级输出（IR / 量化图 / 架构图 / RTL / 报告）均可落盘检查与调试。

### 1.3 支持的输入模型范围（第一版范围）

- **模型形态**：Decoder-only Transformer（GPT 系、LLaMA 系、Mistral 系）为第一目标；
  编码器与部分 RNN 结构列为后续扩展。
- **权重格式**：HuggingFace `safetensors` / PyTorch `.bin` / ONNX，或直接加载 `nn.Module`。
- **精度变换**：输入 fp16/bf16/fp32 → 输出 INT8/INT4/INT2/（可选）fp16 专用核。

---

## 2. 编译器整体架构

编译器采用**层次化流水线（pipeline）**架构，由四个彼此通过**统一 IR** 交换数据的组件（Parser / Quantizer / Architect / RTL Backend）串联而成。
一个组件的内部实现可以自由调整，但只要遵守 IR 契约，就不影响其他组件。

```
  输入: 权重文件 (safetensors/.bin/.onnx) + 配置文件 (config.yaml)
        │
        ▼
╔═══════════════════════════════════════════════════════════════╗
║  Parser  模型解析 · 模型加载与 LLM-IR 生成                     ║
║  目标: 把权重文件 + 模型结构 => 与框架无关的 LLM-IR               ║
╚═══════════════════════════════════════════════════════════════╝
        │  LLM-IR (原始 fp32/bf16 权重 + 静态计算图)
        ▼
╔═══════════════════════════════════════════════════════════════╗
║  Quantizer  量化 · LLM 量化与权重预处理                         ║
║  目标: LLM-IR => 量化后的 QLLM-IR + 权重 ROM 初始化文件           ║
╚═══════════════════════════════════════════════════════════════╝
        │  量化 QLLM-IR + .mem/.coe 权重文件 + 量化元数据
        ▼
╔═══════════════════════════════════════════════════════════════╗
║  Architect  架构 · LLM 硬件架构设计                             ║
║  目标: QLLM-IR => 参数化硬件架构描述 (ArchDesc)                 ║
╚═══════════════════════════════════════════════════════════════╝
        │  ArchDesc (引擎清单、数据流网络、并行度、内存架构)
        ▼
╔═══════════════════════════════════════════════════════════════╗
║  RTL Backend  后端 · RTL 生成与验证部署                           ║
║  目标: ArchDesc => 可综合 RTL + 测试平台 + 比特流/GDS 驱动        ║
╚═══════════════════════════════════════════════════════════════╝
        │
        ▼
  输出: RTL 源码 + 权重 ROM + 测试平台 + 激励 + 资源/性能报告
        (可导入 Vivado / Yosys 综合，最终得到 bitstream / GDS)
```

### 2.1 四层编译器组件职责定义

| 组件 | 文档 | 职责（一句话） | 关键输入 | 关键输出 |
|------|------|--------------|---------|---------|
| **Parser** | `parser.md` | 读懂模型 | 权重文件 + 配置 | `LLM-IR`（计算图 + 权重） |
| **Quantizer** | `quantizer.md` | 压缩数值 | `LLM-IR` | `QLLM-IR` + ROM 文件 + 量化元数据 |
| **Architect** | `architect.md` | 规划电路 | `QLLM-IR` | `ArchDesc`（架构描述） |
| **RTL Backend** | `rtl_backend.md` | 写出电路 | `ArchDesc` | RTL + 测试平台 + 报告 |

---

## 3. 贯穿全流程的中间表示（IR）契约

> IR 是各组件间唯一的**信息交换协议**。全项目统一使用下述数据结构；各组件文档中出现的
> `GraphIR` / `QGraphIR` / `ArchDesc` 均指项目 `llm2asic/ir/` 包下的正式定义。
> Parser / Quantizer / Architect / RTL Backend 不得擅自新增字段而不更新本契约。

### 3.1 三个递进 IR

```
LLM-IR  ──(Quantizer)──▸  QLLM-IR  ──(Architect)──▸  ArchDesc  ──(RTL Backend)──▸  RTL
(fp 权重, 静态图)          (int 权重+scale)         (架构参数+引擎清单)              (文件)
```

1. **LLM-IR**：与框架无关、可序列化的计算图 + 原始浮点权重。由 Parser 产出，Quantizer 消费。
2. **QLLM-IR**：在 LLM-IR 基础上，每个权重张量附加上 `data_q / scale / zero_point / group_config / bit_width`，
   并记录激活量化参数。由 Quantizer 产出，Architect 消费。
3. **ArchDesc**：从"算法层"视角提升到"硬件内核"视角——列出每个引擎（引擎类型、并行度、
   流水线级数、内存位置、权重 ROM 引用）。由 Architect 产出，RTL Backend 消费。

### 3.2 核心数据结构（统一约定）

所有组件统一引用以下 Python 数据类（位于 `llm2asic/ir/`）：

```python
@dataclass
class TensorDesc:
    name: str
    shape: list[int]      # 含所有维度，标量为 []
    dtype: str            # "fp32" | "bf16" | "fp16" | "int4" | "int8" | ...

@dataclass
class WeightDesc(TensorDesc):
    data: np.ndarray | None   # 浮点数据（LLM-IR）或量化数据（QLLM-IR）
    scale: float | np.ndarray | None = None
    zero_point: int | np.ndarray | None = None
    bit_width: int | None = None
    group_size: int | None = None     # 分组量化粒度，0 表示逐张量

@dataclass
class Node:
    name: str
    op_type: str                      # 见 3.3 算子清单
    inputs: list[str]                 # 输入张量名
    outputs: list[str]                # 输出张量名
    attributes: dict                  # 算子特有参数（维度、头数、窗口等）
    weight_names: list[str]           # 引用的权重
    quant: dict | None = None         # 本节点输出的激活量化配置（QLLM-IR 填充）

@dataclass
class GraphIR:                        # 容器：LLM-IR 与 QLLM-IR 共用同一结构
    name: str
    nodes: list[Node]
    tensors: dict[str, TensorDesc]
    weights: dict[str, WeightDesc]
    inputs: list[str]                 # 模型输入（token 流 / 输入嵌入）
    outputs: list[str]
    config: dict                      # 模型级配置（层数、头数、隐藏维……）
```

**不变式（Invariants）**：

- 计算图是**无环（DAG）**且**扁平化**的——子模块被内联展开，不留层级。
- 每个节点的输入/输出张量名必须在 `tensors` 中存在；非权重/非常量的张量具有**静态已知形状**。
- 权重只能通过 `weight_names` 引用，不得内联进 `attributes`（便于量化/重排时统一处理）。

### 3.3 统一算子清单（OP Set）

Parser 负责把上游框架算子（`aten::*` / `onnx::*`）映射到下表的**项目内部算子**；
Architect 据此选择硬件引擎。该表是框架无关抽象的关键。

#### A. 张量/线性算子（GEMM 族）
| 内部算子 | 含义 | 主要 attributes |
|---------|------|----------------|
| `gemm` | 通用矩阵乘法 `C = A@B (+ bias)` | 转置标志 |
| `matmul` | 矩阵乘（含大矩阵分块需要） | — |
| `matvec` / `gemv` | 矩阵-向量乘（**decode 主算子**） | — |
| `bmm` | 批量矩阵乘（attention 的 Q·Kᵀ、·V 用） | batch 维 |
| `linear` | 全连接 + 可选 bias | in/out 特征 |
| `conv2d` | 二维卷积（**非 LLM 主体，保留扩展**） | 核/步长/填充 |

#### B. Attention / 序列算子（LLM 专用）
| 内部算子 | 含义 | 主要 attributes |
|---------|------|----------------|
| `rope` | 旋转位置编码（RoPE） | 维度、基频、theta |
| `attention` | 缩放点积注意力（含 mask/causal） | 头数、头维、因果标志 |
| `kv_store` | KV 缓存写入 | 缓存位置/索引 |
| `kv_load` | KV 缓存读取 | 索引范围 |
| `concat` | 拼接（沿序列/特征维） | axis |
| `unflatten` | 重塑为多头的形状变换 | 目标形状 |

#### C. 归一化 / 激活 / 非线性
| 内部算子 | 含义 |
|---------|------|
| `rmsnorm` | RMSNorm（LLaMA 系标准） |
| `layernorm` | LayerNorm（含 affine） |
| `softmax` | 带温度缩放的 softmax（attention 内） |
| `silu` / `gelu` / `relu` | 常用激活 |
| `add`（残差） | 残差连接（可常量融合） |

#### D. 数据搬运 / 嵌入
| 内部算子 | 含义 |
|---------|------|
| `embedding` | token→向量查表（embedding 矩阵，可固化 ROM） |
| `reshape` / `transpose` / `permute` | 形状/维度变换（常为"视图"，可零成本） |
| `clone` / `copy` | 数据复制（分支） |

> Parser 的解析器遇到**未支持**的算子时，应给出明确的"不支持算子 `xxx`"诊断，
> 并附带建议，而不是静默跳过——这是保证可编译性的前提。

---

## 4. LLM 数据流与硬件形态概览（供 Architect / RTL Backend 参考）

LLM 推理分两阶段，二者对硬件架构的要求截然不同，编译器必须**分别支持**：

1. **prefill（预填充）**：处理一段 prompt，通常一次算很多 token，属于高并行 GEMM / Attention。
2. **decode（自回归逐 token 生成）**：每个新 token 只与全量权重做一次矩阵-向量乘（GEMV），
   并增量更新 KV 缓存；权重是瓶颈，**权重固定带宽**与**KV 缓存读带宽**成为关键。

| 阶段 | 计算形态 | 吞吐 | 内存热点 |
|------|---------|------|---------|
| prefill | GEMM + 批量 attention（bmm） | 高（批/长序列并行） | 权重 + 激活 |
| decode | GEMV + 单 query 注意力 + softmax | token 级流式 | **权重**（全模型都要读一遍/每次 token） |

LLM2ASIC 通过"权重固化在片上"从根本上化解 decode 阶段的权重搬运问题；
KV 缓存的规模则由 `max_seq_len × layers × 2(head_q k/v) × head_dim` 决定，需在编译期做
**容量预算**（见 `quantizer.md` §5 与 `architect.md` §5.3）。

---

## 5. 关键技术决策总览

| # | 决策点 | 本项目推荐 | 理由 / 依据 |
|---|--------|-----------|------------|
| 1 | 框架入口 | `torch.export` 为主，ONNX 为次 | Export IR 扁平化、形状静态；ONNX 覆盖非 PyTorch 来源 |
| 2 | IR 形态 | 扁平化 DAG + 独立权重表 | 便于分析与变换 |
| 3 | 量化策略 | 首层/末层高精度，中间层低比特（INT4/INT2），**分组量化**优先 | LLM 权重对低位更敏感；分组量化显著降误差 |
| 4 | 权重激活方案 | **权重静止（weight-stationary）** + 常量内嵌 ROM | 消除运行时权重加载带宽 |
| 5 | decode 引擎 | 专用 **GEMV 引擎** + 流式 softmax | 匹配 token 级自回归特性 |
| 6 | 层间通信 | AXI-Stream 风格 valid/ready + FIFO | 标准、可综合、易调试 |
| 7 | RTL 生成 | **直接生成 Verilog/SV**（模板化），HLS 作为可选路径 | 本项目面向"权重→RTL"，直接 RTL 最贴合；HLS 用于快速原型 |
| 8 | 验证 | 三层：Python 数值参考 → RTL 仿真（Verilator）→ 上板 | 静态图可全自动交叉比对 |
| 9 | 综合工具 | Vivado（Xilinx/AMD） + Yosys（开源）双目标 | 覆盖商业与开源流程 |
| 10 | 构建确定性 | 全流程无随机、固定 seed、固定输出布局 | 保证可复现性 |

---

## 6. 项目文件与目录结构

```
llm2asic/
├── design/                    # 本设计文档集
│   ├── top.md                 # 本文件（总纲）
│   ├── parser.md              # Parser：模型解析与 LLM-IR
│   ├── quantizer.md           # Quantizer：量化与权重预处理
│   ├── architect.md           # Architect：硬件架构生成
│   └── rtl_backend.md         # RTL Backend：RTL 生成与部署
│
├── llm2asic/                  # 编译器主包（Python）
│   ├── __init__.py
│   ├── cli.py                 # 命令行入口
│   ├── config.py              # 编译配置加载/校验
│   ├── ir/                    # 统一 IR 与算子清单
│   │   ├── ops.py             # Operator/OP 枚举、OP SET 表
│   │   ├── graph.py           # GraphIR/Node/TensorDesc/WeightDesc
│   │   ├── serialize.py        # IR JSON 序列化/反序列化
│   │   └── passes.py           # 邻接/拓扑/算子匹配 工具
│   ├── parser/                 # Parser（读懂模型）
│   │   ├── loader.py          # safetensors/.bin/.onnx/torch.export 加载
│   │   ├── parser.py          # 上游算子 → 内部算子 映射
│   │   └── shape.py           # 形状推导引擎
│   ├── quantizer/             # Quantizer（压缩数值）
│   │   ├── quantizer.py       # 对称/非对称/分组量化
│   │   ├── fuse.py            # RMSNorm/LayerNorm 折叠、scale 折叠
│   │   ├── reorder.py         # 权重重排（GEMV/GEMM 布局）
│   │   ├── rom.py             # .mem/.coe/.h 权重 ROM 生成
│   │   └── kv_plan.py         # KV 缓存规模与分区规划
│   ├── architect/             # Architect（规划电路）
│   │   ├── archdesc.py        # ArchDesc / EngineSpec / DataflowSpec
│   │   ├── gemm_engine.py     # GEMM/GEMV 引擎参数化
│   │   ├── attention.py       # 注意力引擎
│   │   ├── normalize.py       # RMSNorm/LayerNorm/Softmax 引擎
│   │   ├── memory.py          # 内存/带宽/容量预算
│   │   └── dse.py             # 设计空间探索
│   ├── rtl_backend/           # RTL Backend（写出电路）
│   │   ├── verilog.py         # Verilog 代码生成器（模板引擎）
│   │   ├── templates/         # 各算子 RTL 模板
│   │   ├── top_gen.py         # 顶层/数据流网络/控制逻辑生成
│   │   ├── testbench.py       # 测试平台与激励生成
│   │   └── verify.py          # Verilator 仿真与数值比对
│   └── report.py              # 资源/性能/面积报告汇总
│
├── examples/                  # 示例模型与配置
│   ├── gpt2_tiny/             # 小模型端到端 demo
│   └── llama_tiny/            # 精简 LLaMA 结构 demo
├── tests/                     # 单元测试与黄金比对数据
└── pyproject.toml
```

---

## 7. 编译流程（端到端 CLI）

一次典型编译命令：

```bash
llm2asic build \
  --model ./examples/llama_tiny/model.safetensors \
  --config ./examples/llama_tiny/config.yaml \
  --out ./build_out \
  --backend verilog \
  --dse autotune            # 可选：自动设计空间探索
```

编译内部按下列顺序依次调用各组件（管线入口见 `cli.py`）：

```
load_config(config.yaml)
   │
parser.run(model, config)                # Parser:     → LLM-IR
   │
quantizer.run(llm_ir, quant_cfg)         # Quantizer:  → QLLM-IR + ROM 文件
   │
architect.generate(qllm_ir, arch_cfg)    # Architect:  → ArchDesc
   │
rtl_backend.generate(archdesc, out_dir)  # RTL Backend:→ RTL + testbench + 报告
   │
verify.check(rtl_dir, golden_data)       # RTL Backend:→ 通过/失败
```

每条命令都支持 `--dump-ir`、`--dump-arch` 等开关，把中间产物落盘到 `out_dir/<stage>/`，
便于逐级调试。

---

## 8. 技术栈建议

| 层 | 工具 / 库 | 用途 |
|----|----------|------|
| 模型加载 | `torch` + `torch.export` | 首选入口，得到 Export IR |
| 模型加载 | `onnx` + `onnxruntime` | 非 PyTorch 来源 / ONNX 序列化 |
| 权重读取 | `safetensors`、`transformers` | HuggingFace 权重 |
| 数值计算 | `numpy`、`torch` | 量化、参考推演 |
| 图形变换 | 自定义 pass 框架（`ir/passes.py`） | 折叠、重排、资源分配 |
| RTL 生成 | Python 模板引擎（`Jinja2`） | 逐算子 Verilog/SV 生成 |
| RTL 仿真 | `Verilator` + `iverilog` | 开源、快速、可 batch |
| 综合实现 | `Vivado`（商业）/ `Yosys`（开源） | bitstream / 形式化检查 |
| 报告/可视化 | `json` / `graphviz`（可选） | 资源、吞吐、IR 图 |

---

## 9. 研发路线图（Roadmap）

**Phase 0 — 基建（可运行管线）**
- [ ] 统一 IR（`ir/`）与 JSON 序列化
- [ ] CLI 骨架与配置加载
- [ ] `examples/llama_tiny` 端到端"能跑通"最小路径（fp16→fp16，无量化）

**Phase 1 — 数值精度**
- [ ] Quantizer：对称/非对称/分组量化 + RMSNorm/LayerNorm 折叠
- [ ] 三层数值比对框架（Python 参考 ↔ RTL 仿真）
- [ ] 支持 INT8 / INT4 / INT2 权重、混合精度

**Phase 2 — LLM 算子完整化**
- [ ] RoPE、注意力引擎、softmax（流式）、embedding
- [ ] prefill + decode 双调度
- [ ] KV 缓存容量规划

**Phase 3 — 架构与产出**
- [ ] Architect ArchDesc + 设计空间探索
- [ ] RTL Backend 顶层数据流 + 控制逻辑 + 完整 testbench
- [ ] Vivado 综合脚本 + Yosys 支持

**Phase 4 — 规模化与验证**
- [ ] 更多模型结构（Mistral/GPT/更大 LLaMA）端到端
- [ ] ASIC（GDS）流程接入（替代 FPGA 综合）
- [ ] 面积/功耗/吞吐自动化报告

---

## 10. 约束、风险与抵消措施

| 约束 / 风险 | 说明 | 抵消措施 |
|------------|------|---------|
| 静态形状要求 | 动态 shape（动态 head 数、动态维）难以固化 | 编译器对动态维报错并要求 `config.yaml` 固定 |
| 权重位宽越低精度越低 | INT2/INT4 有掉点风险 | 分组量化 + 混合精度 + DSE 权衡 |
| KV 缓存容量大 | 随 seq_len/layers 线性增长 | 编译期预算；支持分片到外部存储 |
| RTL 生成规模大 | LLM 层数多导致顶层庞大 | 模板化 + 参数化复用，Per-layer 实例化 |
| 综合时间 | 复杂数据流综合慢 | 提供模块化综合 + 复用上一次结果缓存 |

---

## 11. 与既有开源生态的对照

| 本项目组件 | 可参考的开源实现 |
|-----------|-----------------|
| Parser（模型解析 / IR） | `torch.export`、ONNX、`qonnx` |
| Quantizer（量化） | 高通 `AIMET`、AMD `Brevitas`、`GPTQ`/`AWQ`（LLM 低位） |
| 图变换 | AMD `FINN`'s `qonnx.transformation` |
| Architect（架构生成） | AMD `FINN`、`NNgen`、`tinyHLS`、`hls4ml` |
| RTL Backend（RTL 生成） | `chipyard`/`rocket` 生成器思想、`vivado` 模板 |
| 综合验证 | `Verilator`、`Yosys`、`iverilog` |

> 本项目不是简单地复用某一家工具链，而是把这些成熟思想收敛到一个**面向 LLM 的、直接产出 RTL** 的
> 编译器里：以"权重固化 + 定制数据流"换取极致能效，同时保持代码规模在单仓库可控范围内。

---

## 12. 总结

`llm2asic` 的定位是：**一个"模型一次性编译、硬件专属定制、权重物理固化"的 LLM→RTL 编译器**。

- 它通过 **Parser → Quantizer → Architect → RTL Backend** 的分层管线，把"读懂模型"→"压缩数值"→"规划电路"→"写出电路"逐步落地；
- 它用一套**贯穿全流程的统一 IR**（`LLM-IR` → `QLLM-IR` → `ArchDesc`）保证各组件松耦合、可独立调试；
- 它以 **LLM 的 prefill/decode 双阶段特性**为核心设计约束，而不是照搬 CNN 数据流模板；
- 最终交付物是**可综合 RTL + 权重 ROM + 测试平台 + 综合脚本**，可被商业或开源工具链直接
  合成为 FPGA 比特流或 ASIC 版图。

以下四份文档（`parser.md` ~ `rtl_backend.md`）分别展开各组件（Parser / Quantizer / Architect / RTL Backend）的详细设计；所有接口、数据结构、
算子清单均以本总纲为基准。
