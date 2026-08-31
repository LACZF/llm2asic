# 模型解析器（Parser）— 模型加载与 LLM-IR 生成

> 组件职责：**读懂模型**。把来自不同框架（PyTorch / ONNX / HuggingFace）的权重与结构，
> 归一化为一套与框架无关、可序列化、静态形状的 **LLM-IR**，供 Quantizer 量化。
> 接口与数据结构以 `top.md` 第 3 节为基准。本组件对应 `llm2asic/parser/`。

---

## 1. 目标与输入输出

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  输入(config):                                                               │
│    --model:  model.safetensors / model.bin / model.onnx / path/to/nn.Module   │
│    --config: config.yaml (dtype、max_seq_len、是否固定 head 数等)               │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Parser: 模型加载 → 算子映射 → 形状推导 → LLM-IR 构建/序列化                    │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  输出:  LLM-IR (GraphIR，见 top.md §3.2)                                      │
│  附带:  out_dir/parser/llm_ir.json (可落盘检查)                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 关键约束（本组件必须保证）
1. **静态形状**：`GraphIR` 中每个中间张量都有确定形状；无法静态确定的输入在解析期报错。
2. **扁平 DAG**：所有 `nn.Sequential`/`nn.ModuleList`/`for` 循环展开为扁平节点。
3. **权重与算子解耦**：权重只经 `node.weight_names` 引用，数值存于 `ir.weights`。
4. **算子闭集**：所有上游算子映射到 `top.md §3.3` 的**内部算子和集**，超出即报"不支持算子"。

---

## 2. 输入格式适配层（Loader）

为隔离不同来源，`parser/loader.py` 暴露统一入口 `load_model(source) -> LoadedModel`，
`LoadedModel` 内部统一暴露 `graph_nodes / initial_params / example_inputs`。

| 来源 | 加载方式 | 说明 |
|------|---------|------|
| `model.yaml` + `.npz` | 纯 NumPy（无需第三方） | **首选**：canonical 权重名 + 声明式配置 |
| `.safetensors` + `config.json` | `safetensors`（numpy 独立加载） | 反序列化权重；结构由 `config.json` 重建 |
| `.onnx` + `config.json` | `onnx.load` | 取 `graph.initializer` 作权重；配置由 `config.json` 重建 |
| `.bin`（PyTorch 状态字典） | `torch.load` | 需 torch；权重为 dict 形式 |
| `.bin`（原始 fp32） + `model.yaml` | 纯 NumPy | exporter 的原始二进制产物 |
| `torch.nn.Module` | `torch.export.export()` | 需 torch；得到扁平 Export IR、状态字典、shape |

所有 `external` 权重（`safetensors` / `onnx` / `bin`）经 `parser/normalize.py`
归一化为统一 canonical 命名（剥离 `model.` 等前缀、映射 `embed_tokens→wte`、
`norm→final_norm`、`GPT-2 的 h.N.ln_*/attn.c_attn` 等），配置由相邻 `config.json`
经 `infer_config` 推导为 Parser 字段。无法映射的权重（如 `o_proj`、位置嵌入、`inv_freq`）
自动忽略，不计入计算图。

### 2.1 首选路径：`torch.export`

```python
import torch
from torch.export import export

model = YourLLM().eval()
example = (torch.randint(0, vocab, (1, seq_len)),)   # token ids

ep = export(model, args=example)
graph = ep.graph_module.graph          # FX 图（扁平化）
sig   = ep.graph_signature             # 参数/权重如何作为图输入
sd    = ep.state_dict                  # 权重数值
```

`torch.export` 的优势：**扁平化**、**静态形状**、**权重提升为图输入**，天然贴合本组件需求。

### 2.2 非 PyTorch / 序列化路径：ONNX

```python
import onnx
from onnx import numpy_helper

m = onnx.load("model.onnx")
onnx.checker.check_model(m)
weights = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
# m.graph.node: 算子节点；node.op_type / node.input / node.output 组成图
```

---

## 3. 算子映射（上游算子 → 内部算子）

核心是一个**映射表**，把 `aten::*` / `onnx::*` 归一到内部算子。映射时同步搬运必要属性
（维度、头数、掩码、因果标志等）。

```python
# parser/parser.py (片段)
OPS = [
    #          上游算子                                   内部算子      属性搬运
    ("aten::mm",                                        "matmul",   {}),
    ("onnx::MatMul",                                    "matmul",   {}),
    ("aten::addmm",                                     "linear",   {"in_features", "out_features"}),
    ("onnx::Gemm",                                      "linear",   {"in_features", "out_features"}),
    ("aten::conv2d", "onnx::Conv",                      "conv2d",   {"kernel", "stride", "padding"}),
    ("aten::rms_norm",                                  "rmsnorm",  {"normalized_shape"}),
    ("aten::layer_norm", "onnx::LayerNormalization",    "layernorm",{"normalized_shape"}),
    ("aten::softmax", "onnx::Softmax",                  "softmax",  {"dim"}),
    ("aten::silu", "onnx::Sigmoid",                     "silu",     {}),   # 注意 sigmoid 配套
    ("aten::gelu", "onnx::Gelu",                        "gelu",     {}),
    ("aten::embedding", "onnx::Gather",                 "embedding",{"num_embeddings","embedding_dim"}),
    ("aten::cat", "onnx::Concat",                       "concat",   {"axis"}),
]
```

> **attention 的识别**：上游常把 attention 拆成 `q@k^T → scale → softmax → attn@v` 的原始算子。
> 本组件应提供 `pattern_match` 把它们**合并为一个 `attention` 节点**（识别 QKV 投影 + 缩放 + causal
> 掩码模式），既便于 Architect 生成专用注意力引擎，也更利于量化处理。合并不了时才保留原始算子级联。

`parser.py` 遍历 `LoadedModel`，逐节点查表并构建 `Node`；节点属性里的**张量形状**
统一由 Shape Engine（§4）在平移后补充。

### 3.1 映射失败的处理

遇到查表未命中的算子：
1. 记录到 `diagnostics.unsupported_ops`，列出算子名、出现位置、可用替代算子；
2. 若该算子所在路径不影响目标（被后续 dead-end 剪枝），可警告继续；
3. 否则**中止并报错**，指出是哪个子模块的哪个算子。

---

## 4. 形状推导引擎（Shape Engine）

`torch.export` / ONNX 一般已带形状，但为稳妥，Parser 内置一个**前向形状推导器**：
从 `inputs` 形状出发，按每个内部算子的规则传播输出形状，直至全图闭合。

```python
def infer_shapes(g: GraphIR) -> None:
    shapes = {name: TensorDesc(name, s, d) for name, s, d in g.inputs_spec}
    topo = topological_order(g)
    for node in topo:
        outs = SHAPE_RULES[node.op_type](node, shapes)   # 传出输出形状
        for out_name, (shape, dtype) in zip(node.outputs, outs):
            shapes.setdefault(out_name, TensorDesc(out_name, shape, dtype))
    # 校验：所有 tensor 都有形状，否则报"无法静态推导形状"错误
```

关键 shape 规则（attention 相关尤其重要）：
- `linear`：`[*in, in_f] → [*in, out_f]`
- `attention`：输入 `[B, S, D]`，head=h，head_dim=d=D/h → 输出 `[B, S, D]`
- `rope`：`[B, S, D]` → `[B, S, D]`
- `softmax`：沿 `dim` 归一化，形状不变
- `kv_store`：写入缓存，形状不变（缓存索引由 attributes 给出）

> **静态化要求**：`S_head`（每头维）与 `h`（头数）必须由输入推导为常量；若模型在运行时
> 动态改变 head 数/维度，则直接判为不可编译，要求用户在配置中显式固定。

---

## 5. 构建与序列化（LLM-IR 落盘）

`parser/exporter.py` 把 `GraphIR` 序列化为 JSON，便于调试、缓存、与后续组件交互。

```json
{
  "name": "llama_tiny",
  "inputs": ["tokens"],
  "outputs": ["logits"],
  "config": {"layers": 6, "hidden": 512, "heads": 8, "head_dim": 64,
             "vocab": 32000, "max_seq_len": 2048},
  "nodes": [
    {"name": "emb", "op_type": "embedding",
     "inputs": ["tokens"], "outputs": ["emb_out"],
     "attributes": {"num_embeddings": 32000, "embedding_dim": 512},
     "weight_names": ["wte"]},
    {"name": "q", "op_type": "linear",
     "inputs": ["hidden0"], "outputs": ["q_out"],
     "attributes": {"in_features": 512, "out_features": 512},
     "weight_names": ["layers.0.q_proj.weight"]},
    {"name": "attn0", "op_type": "attention",
     "inputs": ["q_out", "k_out", "v_out", "pos"],
     "outputs": ["attn0_out"],
     "attributes": {"heads": 8, "head_dim": 64, "causal": true, "scale": 0.125},
     "weight_names": []},
    {"name": "rms0", "op_type": "rmsnorm",
     "inputs": ["hidden0"], "outputs": ["rms0_out"],
     "attributes": {"normalized_shape": [512]},
     "weight_names": ["layers.0.input_layernorm.weight"]}
  ],
  "tensors": {"emb_out": {"shape": [1, 2048, 512], "dtype": "bf16"}, "...": {}},
  "weights": {
    "wte":   {"shape": [32000, 512], "dtype": "bf16", "data_file": "wte.bin"},
    "layers.0.q_proj.weight": {"shape": [512, 512], "dtype": "bf16", "data_file": "..."}
  }
}
```

### 5.1 权重外置（weight 与 IR 分离）
为避免超大 JSON，权重数值不内联进 IR 文件，而是：
- 每个权重导出一个 `*.bin`（numpy 原始二进制），
- IR 中用 `data_file` 字段引用路径，
- 同时生成 `manifest.json` 列出所有 `weight_name → 文件 + 形状 + dtype`
  （Quantizer 据此逐张量量化和重排）。

---

## 6. 前端代码结构

```
llm2asic/parser/
├── loader.py        # load_model()：多来源统一加载
├── normalize.py     # external 权重/配置 → canonical 命名与字段
├── parser.py        # 上游算子 → 内部算子；attention 模式匹配
├── shape.py         # Shape Engine (infer_shapes)
├── builder.py       # 组装 GraphIR / 拓扑排序 / 死代码剪枝 / 校验
└── exporter.py      # LLM-IR JSON + 权重 bin + manifest 落盘
```

`builder.py` 在 GraphIR 构建后统一执行**校验 pass**：
- 有环检测
- 权重引用完整性
- 所有张量形状已推导
- 所有算子属于内部和集
- 输出端存在（logits 可达）

---

## 7. 与下游组件的接口约定

Parser 产出交给 Quantizer 的内容：

| 产物 | 路径 | 用途 |
|------|------|------|
| `llm_ir.json` | `out/parser/` | 计算图 + shape（Quantizer 读取） |
| `weights/*.bin` | `out/parser/weights/` | 原始浮点权重（Quantizer 量化） |
| `manifest.json` | `out/parser/` | 权重清单（Quantizer 遍历入口） |
| `diagnostics.json` | `out/parser/` | 不支持算子 / 警告汇总 |

> 本组件不引入任何硬件相关概念（并行度、位宽、ROM 等），那是 Quantizer / Architect 的职责。

---

## 8. 健壮性与可扩展性

- **算子集可扩展**：新增算子 = 在 `OPS` 表加一行 + `SHAPE_RULES` 加一条 + 映射 fallback。
  无需改动框架骨架。
- **错误可读**：所有编译错误带"文件/节点/建议"三重信息，便于定位模型中的问题算子。
- **元数据保留**：节点上保留 `source`（上游框架算子名与源码位置），便于与 RTL 对应调试。
- **缓存友好**：LLM-IR 可序列化，二次编译直接加载缓存跳过解析。

### 总结

Parser 的输出是一份**干净、静态、可序列化**的 `LLM-IR`：它抹平了 PyTorch/ONNX/HuggingFace
的差异，把任意上游模型归一为统一算子图 + 权重表，并提前暴露了 attention/embedding/rope 等
**LLM 特有结构**（以独立 `attention` 节点体现），为 Quantizer 的量化与 Architect 的专用引擎
生成扫清了障碍。
