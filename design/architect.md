# 架构规划器（Architect）— LLM 硬件架构生成

> 组件职责：**规划电路**。把 Quantizer 的 `QLLM-IR` + 量化元数据 + KV 规划，综合成一套
> **参数化硬件架构描述（ArchDesc）**——列出每个硬件引擎的类型、并行度、流水级数、内存位置、
> 权重 ROM 引用与层间数据流网络。ArchDesc 是后续 RTL 后端生成 RTL 的唯一蓝图。
> 接口/数据结构以 `top.md` §3 为基准。对应 `llm2asic/architect/`。

---

## 1. 目标与输入输出

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  输入:  QLLM-IR (qllm_ir.json)                                                │
│         + quant_metadata.json (量化布局/scale/ROM)                            │
│         + kv_plan.json (KV 缓存布局)                                          │
│         + arch_cfg.yaml (目标器件、时钟、并行度、设计空间探索开关)               │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Step 1  算子 → 引擎映射 与 计算图降维                                          │
│  Step 2  引擎原语设计     (GEMV/GEMM、Attention、Softmax、RMSNorm、Embedding)  │
│  Step 3  数据流与内存架构  (prefill/decode 双调度、KV 缓存、FIFO/带宽)          │
│  Step 4  设计空间探索     (DSE：并行度 × 位宽 → 面积/吞吐权衡)                  │
│  Step 5  输出 ArchDesc    (ArchDesc JSON)                                      │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  输出:  ArchDesc (架构描述 JSON) → 交给 RTL Backend 生成 RTL                      │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. LLM 推理的两种硬件形态（本组件最核心的洞察）

LLM 推理不是单一数据流，而是**两个截然不同的阶段**。Architect 必须分别建模，RTL Backend 生成
的 RTL 要能**复用一个可重构/时分复用**的引擎集合同时覆盖两阶段。

```
prefill（高并行 GEMM）            decode（token 级 GEMV）
─────────────────────             ─────────────────────
一次算 S 个 token               一次算 1 个新 token
Q·Kᵀ / ·V：bmm(G, S, S)         q 与全量 K/V 做 (1,S) 内积
权重利用率高                      权重每 token 全读一遍(瓶颈)
激活内存大                        权重带宽是唯一瓶颈（权重已在片上 ROM）
```

### 2.1 设计取舍：一个引擎集合 + 双模式
- 同一个 **GEMV/GEMM 引擎**，通过 `mode` 信号在"矩阵乘-向量"与"矩阵乘-向量批量"间切换：
  - prefill：按行流式读入多个 token（GEMM）
  - decode：单 query，权重静止，跨全行流水（GEMV, weight-stationary）
- 同一个 **Attention 引擎**处理两种注意力的内积（见 §4）。prefill 时 K/V 从片上读全量，
  decode 时增量写 + 读全量。
- **Softmax** 采用**流式在线 softmax**（见 §4.6），与 token 到来同步，避免存储全部注意力分数。

---

## 3. Step 1：算子 → 引擎映射

`arch/archdesc.py` 定义两类规格：

```python
@dataclass
class EngineSpec:          # 一个硬件引擎实例
    engine_type: str       # gemm_engine | attention_engine | softmax_engine |
                          # rmsnorm_engine | embedding_engine | ...
    pe: int                # 输出并行度
    simd: int              # 输入并行度
    pipeline_stages: int   # 流水级数
    weight_rom: str|None   # 绑定的权重 ROM 引用
    attrs: dict

@dataclass
class DataflowLink:
    src: str               # 源引擎
    dst: str               # 目标引擎
    fifo_depth: int
    data_width: int
    protocol: str = "axis" # valid/ready
```

映射表（从内部算子 → 引擎，对应 `top.md §3.3` 的内部算子集）：

| 内部算子 | 引擎 | 说明 |
|---------|------|------|
| `linear` / `matmul` / `matvec` / `gemm` | `gemm_engine` | 统一"矩阵向量"引擎，双模式 |
| `attention`（prefill bmm 部分） | `attention_engine`（计算内核） | Q·Kᵀ、·V |
| `softmax` | `softmax_engine`（流式） | 在线 softmax |
| `rope` | `rope_engine`（可并入 attention 前处理） | RoPE 旋转 |
| `rmsnorm` / `layernorm` | `rmsnorm_engine` | 归一化 |
| `silu` / `gelu` / `relu` | `activation_engine`（LUT/算术） | 激活 |
| `embedding` | `embedding_engine` | token→向量查表（ROM） |
| `kv_store` / `kv_load` | 内存/缓存引擎（`memory`） | KV 缓存访问 |
| `add`（残差） | `vector_op_engine` | 逐元素 |

> **共享 vs 专用**：默认让多个相同算子**共享引擎实例**（减少面积），通过 ArchDesc 的
> `instance_of` 引用复用。是否共享由 DSE 决定（见 §6）。

---

## 4. Step 2：核心引擎原语

### 4.1 GEMV/GEMM 引擎（decode 重中之重）

架构基于"**权重静止（weight-stationary）**"：权重常驻 PE 阵列旁或 ROM，输入 token 流动。

```
              ┌─────────────────────────────────────────────┐
   in[1,C_in]  │           GEMV Engine (PE×SIMD)            │
  ────────────▶│  ┌──────┐ ┌──────┐        ┌──────┐          │
              │  │ PE 0 │ │ PE 1 │  ...   │ PE P │          │
              │  └──────┘ └──────┘        └──────┘          │
              │   每个 PE = SIMD 个乘法 + 累加器               │
              │   权重 ROM 每周期提供 PE×SIMD 个权重            │
              │   ┌──────────────────┐                        │
              │   │  scale 定点化(乘/移位) + 偏置 + 截断        │
              │   └──────────────────┘                        │
              └───────────┬─────────────────────────────────┘
                          ▼
              out[1,C_out]
```

**参数**：`pe`（输出并行）、`simd`（输入并行）、`acc_width`（累加位宽 = 32）、
`mode`（gemm/gemv）。**读地址生成器**按 `quant_metadata.layout` 生成连续地址，从
`weights_rom` 顺序读打包权重并拆包分发到 PE。

**定点 greedy**：
```
acc += (q_w * x) ; 每个分组末尾: acc = ((acc * s_mant) >> s_shift)
```

这种引擎一次覆盖 `linear/matmul/matvec` 全部权重重心，且 decode 时是核心。

### 4.2 Attention 引擎

对每个 query，计算与所有 K 的注意力分数，再对 V 加权求和。**decode 下为 GEMV 形态的多头
内积**，prefill 下为 bmm。

```
q[1,hd] · K[hd, S] → scores[1,S]     (每个 key head 一列 → 点积)
scores[1,S] → softmax → attn[1,S]
attn[1,S] · V[S,hd] → out[1,hd]
```

引擎内在流水：
1. **QK 点积单元**：读 K 缓存（来自 kv_plan），与 q 做内积，得 `score_i`
2. **在线 softmax**（§4.6）累积得 `softmax(score_i)`
3. **AV 加权**：把 `softmax(score_i) · v_i` 累加到输出，并随 softmax 迭代修正
4. causal mask：`i > 当前 token 位置` 的 score 强制为 `-inf`

> 硬件复用：第 1 步就是 `gemm_engine` 的一种调用（q 向量 × K 矩阵）；第 3 步是
> `attn 向量 × V 矩阵`。因此 **Attention 引擎是对 gemm_engine 的封装** + softmax + mask，
> 面积可控。

### 4.3 RoPE 引擎
对 q/k 向量按 Rotary 位置编码旋转：`(x_rot) = [x;-x] @ [cos;sin]`，可由 2×2 旋转变换 + 三角
查找表实现。decode 时每个新位置只需读当前位置的 cos/sin。

### 4.4 RMSNorm / LayerNorm 引擎
流式计算：先算 `rms(x)`（或 mean/var），再除并以 gamma 缩放。可拆为"规约 + 逐元素缩放"两个
微级，能耗低、简单。若 Quantizer 已折叠则无需独立引擎。

### 4.5 Embedding 引擎
`token_id → row` 的查表。embedding 矩阵固化 ROM，按 token 索引读出向量，送入 GEMV。
支持共享绑定（tied embedding 时 output_proj 复用同一 ROM）。

### 4.6 流式 Softmax（在线 softmax）

为兼容 token 流式到达，使用**在线 softmax**：不缓存全部 S 个 score，而是维护运行最大
`m` 与运行和 `sum`：

```python
m = -inf; sum = 0
for i in range(S):
    m_new = max(m, s_i)
    f = exp(m - m_new)
    sum = sum * f + exp(s_i - m_new)
    m = m_new
    # 输出概率仍要等 finish 再整体除 —— 硬件用两遍或保留修正项
```

硬件实现两种：
- **two-pass**：一遍求 m/sum，遍二输出 `softmax(s_i)`（需暂存 score 或重算）
- **此场景推荐**：由于 attention 输出是 `Σ softmax_i · v_i`，可把 softmax 分母带入 AV 累加，
  在单遍内完成（"flash-attention 式"累积 + 重缩放），避免两次访问 K/V。

---

## 5. Step 3：数据流与内存架构

### 5.1 ArchDesc 的数据流图
ArchDesc 含一张**引擎级数据流图**（顶点=引擎实例，边=`DataflowLink`），与 LLM-IR 的算子图
不同——它已经"实现化"：包含共享引擎、FIFO、控制器、KV 内存分区。

```
[mode: prefill|decode]
tokens ─▶ embedding ─▶ rmsnorm ─▶ gemm(q) ─▶ rope ─┐
                          │                          │
                          ▼                          ▼
                     gemm(k)─▶rope─▶K缓存─┐      attention.qk ─▶ softmax ─▶ attention.av
                     gemm(v)────────────▶V缓存─┤                              │
                                               ▼                              ▼
                                          [残差 add] ◀──── rmsnorm2 ── gemm(up/gate) ── silu
                                               │
                                               └───▶ gemm(down) ─▶ ...
```

### 5.2 prefill/decode 双调度
`archdesc.json` 里为一个共享引擎集合声明**两种调度表（schedule）**：
- `prefill_sched`：高并行 GEMM，token 批处理，attention 用 bmm
- `decode_sched`：GEMV weight-stationary，单 query，KV 增量更新

RTL Backend 据此生成一个**顶层控制器**（FSM），根据 `run_mode` 输入在两条流水之间切换。

### 5.3 内存与带宽预算
- **权重 ROM**：全部在片上（权重静止），大小 = 各量化权重 ROM 之和（来自 Quantizer 估算）
- **KV 缓存**：按 `kv_plan.json` 分块，标出每块位置（BRAM/URAM/external），计算所需读带宽
  （decode 时每 token 读全部 K 的带宽若超标，提示提高并行或外置降级）
- **激活 FIFO**：层间缓冲深度由"上游产生速率 vs 下游消费速率"推导，至少覆盖流水停顿

`arch/memory.py` 先做**预算检查**：
```
片上总需求(权重ROM + KV 片上部分 + FIFO) <= 目标器件片上资源
超出或带宽超标 → 报错并建议调整（增大并行 or 外置 KV or 升级型号）
```

---

## 6. Step 4：设计空间探索（DSE）

`arch/dse.py` 在"并行度 × 位宽"空间内搜索满足资源限制的最优配置，目标函数 = 代价
（面积）以达成目标吞吐。

```python
def explore(qllm_ir, device, objective="throughput_per_area"):
    for pe in [8, 16, 32, 64]:
        for simd in [4, 8, 16]:
            for mode in ["shared", "dedicated"]:
                res  = estimate_resources(pe, simd, mode)   # LUT/DSP/BRAM/URAM
                thru = estimate_throughput(qllm_ir, pe, simd, mode)
                if within_limits(res, device):
                    candidates.append(score(pe, simd, mode, res, thru))
    return top_k(candidates)
```

资源估算模型（初版简化）：
```
GEMV 引擎:  LUT ≈ pe*simd*K_lut + pe*simd*K_acc,  DSP ≈ pe*simd(乘法器)
Attention: 额外 + 在线 softmax 的 LUT 分量
Embedding/ROM: BRAM 用量
```

> 为控制综合时间，DSE 先用**解析模型**粗筛，命中前 N 个候选后再让 RTL Backend 仅对这些候选
> 做精细估算/综合。

---

## 7. Step 5：输出 ArchDesc（规范）

`archdesc.json` 结构示例：

```json
{
  "top_module": "llm_accel",
  "input_width": 8, "output_width": 8,
  "users_facing_interfaces": {"in": "axis", "out": "axis", "mode": "ap_ctrl"},
  "engines": [
    {"id": "gemm_0", "type": "gemm_engine", "instance_of": "shared_gemm",
     "pe": 16, "simd": 8, "pipeline_stages": 3, "weight_rom": "q_proj_weight.mem",
     "scale_rom": "q_proj_scale.mem", "attrs": {"modes": ["gemm","gemv"]}},
    {"id": "attn_0", "type": "attention_engine", "pe": 8, "heads": 8, "head_dim": 64,
     "causal": true, "kv_partition": "onchip", "k_rom": "...", "v_rom": "..."},
    {"id": "softmax_0", "type": "softmax_engine", "online": true, "stages": 2},
    {"id": "rms_0", "type": "rmsnorm_engine", "pe": 16, "scale_rom": "layernorm0.mem"},
    {"id": "emb_0", "type": "embedding_engine", "rom": "wte_weight.mem", "vec_width": 512},
    {"id": "mem_kv_0", "type": "kv_memory", "onchip_bytes": 6000000, "partitions": [...], "external": []}
  ],
  "dataflow": [
    {"src": "emb_0", "dst": "rms_0", "fifo_depth": 2048, "data_width": 64},
    {"src": "gemm_0(q out)", "dst": "attn_0", "fifo_depth": 4096, "data_width": 64},
    ...
  ],
  "schedulers": {
    "prefill": [["emb_0"],["gemm_0","gemm","batch"],["attn_0","bmm"],["softmax_0"],["attn_0","av"]],
    "decode":  [["emb_0"],["gemm_0","gemv"],["attn_0","single"],["softmax_0"],["attn_0","av"]]
  },
  "resource_estimate": {"LUT": 12450, "DSP": 384, "BRAM": 16, "URAM": 0},
  "target_device": "xczu7ev-ffvc1156-2-e",
  "schedules_note": "顶层 FSM 依据 mode 切换 prefill/decode 流水"
}
```

### 组件边界
- Architect **只产出 ArchDesc + 资源/性能估算**，**不产出任何 RTL**；
- RTL 生成、综合、仿真、部署全部是 RTL Backend 的职责；
- 因此 Architect 的所有引擎描述必须**可被 RTL Backend 模板直接实例化**（引擎类型集合与
  RTL Backend 的 RTL 模板一一对应，任何新增引擎类型，RTL Backend 必须有对应模板）。

---

## 8. 健壮性与可扩展性
- **引擎模板化**：每种 `engine_type` 对应模板 + 参数校验器；新增引擎 = 加一个类型。
- **资源预算硬门禁**：超出目标器件资源立即报错，避免生成不可综合的设计。
- **DSL 可序列化**：ArchDesc 是纯 JSON，可落盘、可 diff、可版本化。
- **离线可分析**：吞吐/面积/带宽的估算模型与实现解耦，便于迭代校准。

### 总结
Architect 把"LLM 算法图"转化为"可实例化的硬件引擎 + 数据流 + 双模式调度 + 内存预算"的
**ArchDesc**。它抓住 LLM 的 prefill/decode 双阶段特性，以"权重静止的 GEMV 引擎 + 流式
在线 softmax + KV 缓存规划 + 共享引擎时分复用"为核心手段，在可控面积下达成目标吞吐，
为 RTL Backend 的 RTL 落地提供唯一且充分的蓝图。
