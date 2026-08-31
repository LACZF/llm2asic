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
