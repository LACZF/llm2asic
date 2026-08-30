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

输入模型文件：`examples/llama_tiny/model.yaml` 声明式模型描述（权重由示例脚本生成），
或任意 `torch.nn.Module`（若安装了 torch）。
