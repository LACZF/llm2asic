# 量化器（Quantizer）— LLM 量化与权重预处理

> 组件职责：**压缩数值**。把 Parser 的 `LLM-IR`（浮点权重）量化为低位定点，进行针对
> LLM 硬件形态（GEMV/GEMM、权重静止、KV 缓存）的预处理与重排，最终产出 **QLLM-IR** +
> 权重 ROM 初始化文件 + 量化元数据。接口/数据结构以 `top.md` §3 为基准。对应 `llm2asic/quantizer/`。

---

## 1. 目标与输入输出

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  输入:  LLM-IR (llm_ir.json + weights/*.bin + manifest.json)                  │
│         + 量化配置 (quant.yaml: 策略/位宽/分组/混合精度)                        │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Step A  图优化与算子融合   (RMSNorm/LayerNorm 折叠、scale 折叠)                │
│  Step B  权重量化          (对称/非对称、逐张量/逐通道/逐分组、混合精度)         │
│  Step C  权重预处理与重排  (GEMV/GEMM 布局、打包、稀疏处理)                     │
│  Step D  KV 缓存规划       (容量预算、分片决策)                                 │
│  Step E  权重 ROM 生成     (.mem/.coe/.inc)  + 量化元数据                       │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  输出:  QLLM-IR (量化后 GraphIR) + weights_rom/* + quant_metadata.json        │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 与 CNN 编译器（FINN 等）的关键差异

LLM 的量化难点与重点和 CNN 不同：

| 方面 | CNN（FINN 习惯） | **LLM（本项目重点）** |
|------|-----------------|----------------------|
| 输入尺度 | 图片（0~1/0~255） | token 嵌入、连续向量；**激活含异常大值（outlier）** |
| 计算瓶颈 | GEMM（prefill/卷积） | decode 的 **GEMV**；权重是唯一瓶颈 |
| 量化粒度 | 常逐张量/逐通道 | **逐分组（group=128）** 精度收益大 |
| 权重分布 | 相对均匀 | 大模型权重常近似高斯，低位 RMS 误差敏感 |
| 特殊算子 | — | attention 的 `softmax`、`rope`、`rmsnorm` 需专门处理 |

因此本组件不仅做"量化"，还引入 **per-group 量化 + outlier 感知 + KV 缓存规划**，这是
LLM-to-RTL 能落地的关键。

---

## 2. Step A：图优化与算子融合

量化前先做图优化，能显著降低硬件复杂度与量化误差。

### 2.1 RMSNorm / LayerNorm 折叠

LLaMA 系残差结构 `y = x + RMSNorm(x) · W` 中，`RMSNorm` 的权重 `γ` 可与后续方阵融合，
减少一个中间乘。折叠原则：**只折叠不引入额外误差**的线性变换；`softmax`、`silu` 等非线性
保持独立。

```python
def fold_rmsnorm_into_linear(weight, gamma, eps, rmsnorm_mode=True):
    if rmsnorm_mode:
        # RMSNorm: out_norm = x / rms(x) * gamma
        # 这里的 gamma 不直接折叠进线性层(W)，而是线性层输出端除 rms(x)。
        # 更实用的折叠: 把 gamma 并入下游 GEMM 的行缩放。
        return weight * gamma.reshape(-1, 1)   # (out,in) 每行乘 gamma
    # LayerNorm: out = (x - mean)/std * gamma + beta, 折叠进 W/b
    scale = gamma / std
    return weight * scale.reshape(-1, 1), (bias + mean*... )  # 并入偏置
```

### 2.2 激活量化 scale/zero-point 折叠

若激活也量化（INT8 激活），则 quantize/requantize 的 scale 尽量合并到相邻的乘法/加法定点
运算中，避免硬件里频繁乘除法。原则：**把 scale 尽可能折叠为硬件可预先计算的常数**
（Architect 会据此生成"带 scale 的乘累加"）。

### 2.3 融合的收益
- 减少一个串行算子（少一级流水、少一个 FIFO）
- 减少一次 round 引入的误差累积
- 降低顶点延迟

---

## 3. Step B：权重量化

### 3.1 量化方案矩阵

| 方案 | 形式 | 适用 | 硬件成本 |
|------|------|------|---------|
| 对称逐张量 | `q = round(w / s)`，s 单标量 | 权重近似 0 对称 | 最低 |
| 对称逐通道 | 每输出行一个 s | 通道分布差异大 | 低（s 存 ROM） |
| 对称逐分组 | 每 G=128 个权重一个 s | **LLM 主流，精度高** | 中（G s 存 ROM） |
| 非对称 | 带 zero_point | 激活用（如 softmax 前） | 中 |
| 更低比特 | INT2 / 三元 / 二进制 | 极致压缩（实验） | 高精度风险 |

> **LLM 推荐默认**：权重用**逐分组对称量化**（group=128），激活用 INT8 非对称（若需激活量化）。
> `embedding` 与 `output_proj`（embedding 共享权重）对精度敏感，可单独配置更高位宽。

### 3.2 分组量化实现（核心）

```python
def quantize_group(w, bit_width, group_size, sym=True):
    """对 [C_out, C_in] 权重按每 group_size 个连续权重为一组进行量化。
    返回 q(量化整型), scale, zero_point, shape 便于重排。
    """
    c_out, c_in = w.shape
    w2 = w.reshape(c_out, -1, group_size)          # [C_out, G, group]
    if sym:
        amax = w2.abs().amax(dim=-1, keepdim=True)
        scale = amax / (2**(bit_width-1) - 1)
        q = torch.round(w2 / scale).clamp(-2**(bit_width-1), 2**(bit_width-1)-1)
        return q.to(torch.int8), scale.squeeze(-1), torch.zeros_like(scale.squeeze(-1))
    # ...非对称类似
```

**硬件影响**：dequant 在硬件里是 `acc += q_w * x * s_w(i)`。逐分组意味着每个乘累加后要乘以
分组 scale——Architect 会把 scale 重排成与权重匹配的 ROM 布局，并在 PE 输出端做定点化。

### 3.3 outlier（异常大值）感知
LLM 激活/权重常有少量大幅值（outlier）。可选策略：
- **LLM.int8()/Outlier-aware**：把 outlier 通道拆分到高精度路径（如单独 fp16 GEMM）。
- **本项目轻量方案**：对 outlier 集中的**通道/分组**单独提高量化位宽（混合精度的一种特例）。

bool 决定：默认不做矩阵分解（避免面积爆炸），仅在误差超阈时对**特定分组**升级位宽。

### 3.4 混合精度配置（YAML 示例）

```yaml
quant:
  default_weight: {scheme: "symmetric_group", bit_width: 4, group_size: 128}
  exceptions:
    "wte":                 {scheme: "symmetric", bit_width: 8}      # embedding 高精度
    "output_proj.weight":  {scheme: "symmetric", bit_width: 8}      # 末层/共享层
    "layers.0.*.weight":   {scheme: "symmetric_group", bit_width: 8} # 首层高精度
  activation:              {scheme: "symmetric", bit_width: 8}
  target_metric_deg: 0.05   # 允许的最大数值性能退化，超限则告警/升级位宽
```

### 3.5 精度验证（量化质量门禁）
量化后必须跑**黄金比对**：用 `examples/` 里的小模型与少量校准输入，对比
`fp16 参考输出` 与 `量化模拟输出`（在 torch/numpy 里模拟定点），计算：
- 每 token 的均方误差 / 余弦相似度
- 顶层 logits 的 top-1 命中率退化

不达标时：自动尝试升级敏感层位宽（简单版 DSE）或提示用户调整 `quant.yaml`。

---

## 4. Step C：权重预处理与重排

LLM 硬件以 **GEMV/GEMM** 为核心，权重布局要匹配 PE/SIMD 数据流与"权重静止"策略。

### 4.1 GEMV 权重布局（decode 主用）
decode 是一个 `[1, C_in] x [C_in, C_out] → [1, C_out]` 的矩阵向量乘。硬件把 `C_out` 拆成
`PE` 个并行输出、`C_in` 拆成 `SIMD` 个并行输入。重排目标：**每个周期，PE×SIMD 个权重**
一次性从 ROM 读出，供 PE 阵列使用。

```python
def reorder_gemv(w_q, scale, pe, simd, group):
    # w_q: [C_out, C_in] 分组量化后
    C_out, C_in = w_q.shape
    # pad 到 PE / SIMD 整数倍
    w  = pad(w_q,  (C_out % pe), (C_in % simd))
    # 重排为 [C_out/pe, C_in/simd, pe, simd]，即 PE 块在外、SIMD 块在内
    w  = w.reshape(C_out//pe, pe, C_in//simd, simd)
    w  = w.transpose(0, 2, 1, 3)          # [c_out_blk, c_in_blk, pe, simd]
    # scale 独立数组或与权重交错存储
    s  = reorder_scale(scale, pe, group)  # 与 w 对齐
    return w, s
```

### 4.2 低位打包（packing）
`bit_width < 8` 时多个权重打包进一个存储字，减小 ROM 位宽与带宽：
- INT4：2 个/字节；INT2：4 个/字节。
- 打包顺序与 PE 输入拆包顺序一致（Architect 的 GEMV 引擎按此布局拆包）。

```python
def pack_lowbit(q, bit_width):
    per_byte = 8 // bit_width
    flat = q.reshape(-1)
    flat = pad(flat, to_multiple_of(per_byte))
    packed = []
    for i in range(0, len(flat), per_byte):
        b = 0
        for j, v in enumerate(flat[i:i+per_byte]):
            b |= ((v & ((1<<bit_width)-1)) << (j*bit_width))
        packed.append(b)
    return np.array(packed, dtype=np.uint8)  # 每元素1字节，位宽打包
```

### 4.3 稀疏 / 剪枝
若模型已剪枝（大量 0 权重），可选择性启用稀疏编码（CSR 风格 index+value），在 GEMV 中跳过
全零组。**默认关闭**——权重固化后存储紧凑已足够，稀疏化需权衡控制逻辑复杂度。此项保留为
开关 `sparse: false`。

### 4.4 偏置与 scale 的定点化
- 量化后偏置：`b_q = round(b / (s_w * s_x))`，存储为 32-bit 定点，供累加器直接相加。
- 分组 scale：`s_w` 转成整数 + 移位表示 `(mantissa, exponent)`，供硬件定点乘法/移位。

---

## 5. Step D：KV 缓存规划

attention 的 K/V 缓存随 `seq_len × layers` 线性增长，是 LLM 片上内存的最大开销之一。

### 5.1 容量公式
```
KV_bytes = layers × 2 × head_dim × seq_len × dtype_bytes
         （只算单 query 头；K、V 各一份）
示例: 6 层, 8 头, head_dim=64, seq=2048, int8
    = 6 × 8 × 64 × 2048 × 2 × 1 byte
    = 6 × 8 × 64 × 2048 × 2 = 12,582,912 B ≈ 12 MB/头对
```
因此**KV 缓存必须按层 × 头 分块**，并决定哪些放片上 BRAM/URAM、哪些外置到 DRAM。

### 5.2 规划输出（交给 Architect）
`kv_plan.json` 描述：
- 每层 K/V 的存储位置（on-chip / external）
- 是否按头分块、块内地址映射
- ping-pong 缓冲策略（decode 增量写 + 全量读的 overlap）
- 当 `KV_bytes` 超过片上预算时的降级方案（如只用少数头在片上、其余外置 + 带宽评估）

```python
def plan_kv(cfg) -> dict:
    per_layer = 2 * cfg.heads * cfg.head_dim * cfg.max_seq * bytes_per_elem
    total = per_layer * cfg.layers
    onchip_cap = cfg.onchip_mem_bytes
    # 简单策略：能全放片上则全片内；否则按层平均分配到片上/片外
    ...
    return {"per_layer_bytes": per_layer, "total_bytes": total,
            "onchip_layers": n, "external_layers": cfg.layers - n,
            "external_bw_req_GBs": estimate_bw(...)}
```

---

## 6. Step E：权重 ROM 生成

量化、重排、打包后的权重写入 ROM 初始化文件。

### 6.1 格式

| 格式 | 用途 | 备注 |
|------|------|------|
| `.mem` | Verilog `$readmemh` 通用 | 每行一个十六进制字，最通用 |
| `.coe` | Vivado Block Memory | 商业工具兼容 |
| `.inc` | C/C++ 或模板内嵌 | HLS/常量数组内嵌（可选路径） |

统一由 `quantizer/rom.py::ROMGenerator` 生成；每个权重张量一个文件，文件名 = 量化后的
规范化层名。

### 6.2 元数据（quant_metadata.json）
每个量化权重记录：`scale/zero_point/bit_width/group_size/布局/ROM 文件/地址`——
Architect / RTL Backend 依据它生成 PE 拆包与 scale 定点逻辑，并生成量化模拟基准用于验证。

```json
{
  "q_proj.weight": {
    "scheme": "symmetric_group", "bit_width": 4, "group_size": 128,
    "scale_rom": "q_proj_scale.mem", "scale_is_int": true,
    "scale_mantissa_bits": 8, "scale_exponent_bits": 8,
    "rom_file": "q_proj_weight.mem", "rom_depth": 65536, "rom_width": 32,
    "layout": {"pe": 16, "simd": 8, "reordered_shape": [32, 64, 16, 8]}
  },
  "k_v_plan": { ... }   // 见 Step D
}
```

### 6.3 输出目录
```
out/quantizer/
├── qllm_ir.json                 # 量化后的 GraphIR（QLLM-IR）
├── weights_rom/                 # 所有权重 ROM
│   ├── q_proj_weight.mem
│   ├── q_proj_scale.mem
│   ├── wte_weight.mem
│   └── ...
├── quant_metadata.json          # 量化元数据（Architect 入口之一）
├── kv_plan.json                 # KV 缓存规划
└── quant_report.json            # 精度回退对比/资源估算报告
```

---

## 7. 与下游组件（Architect / RTL Backend）接口约定

Quantizer 产出物：

| 产物 | 说明 | 消费方 |
|------|------|--------|
| `qllm_ir.json` | 量化图：每个权重带 scale/zp/bit_width/group/layout | Architect |
| `weights_rom/*` | 打包后权重 ROM | 复制到 RTL Backend 输出 |
| `quant_metadata.json` | 拆包/scale 定点/地址信息 | Architect 与 RTL Backend |
| `kv_plan.json` | KV 缓存布局与带宽 | Architect |
| `quant_report.json` | 精度回退验证结果 | 编译报告 |

> 约定：**ROM 布局（排列与打包）与本组件固定**，Architect 的 GEMV/GEMM 引擎与 RTL Backend 的
> RTL 必须按 `quant_metadata` 的 `layout` 字段一致地拆包读取。这是跨组件耦合力最强的点，
> 请务必先在 `top.md` §3 的 IR 契约（由 Parser 产出的 `LLM-IR`）上统一形状与字段名，避免布局不同步。

---

## 8. 健壮性与可扩展性
- **量化方案矩阵化**：新增方案只需在 `quantizer.py` 注册 `(scheme, bit_width, group)` 处理函数。
- **布局可参数化**：`reorder.py` 完全由 `pe/simd/group` 参数驱动，与具体算子解耦。
- **精度门禁可配置**：不达标自动升级位宽或中止，保护生成 RTL 的可用性。

### 总结
Quantizer 的核心是"**在保证精度的前提下，把 LLM 权重压到能塞进片上 ROM 并支持高效 GEMV
流式读取**"。通过逐分组量化、Outlier 感知、GEMV 重排、低位打包与 KV 缓存规划，它产出
`QLLM-IR + ROM + 元数据`，让 Architect 得以针对"权重静止的 decode 引擎"展开架构设计。
