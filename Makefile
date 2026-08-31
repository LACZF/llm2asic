# LLM2ASIC Makefile
#
# 在当前目录直接执行：
#   make            # 先跑测试，再生成 RTL 并仿真验证（默认目标）
#   make test       # 仅运行 pytest 测试套件
#   make rtl        # 端到端：生成 RTL + 黄金参考 + 仿真逐位比对
#   make synth      # Yosys 综合，产出最终门级网表 netlist.v（需 yosys）
#                   #   BACKEND=fpga       -> synth_xilinx（默认）
#                   #   BACKEND=asic PDK=sky130hd LIBERTY=xxx.lib -> 标准单元网表
#   make clean      # 清理构建产物与缓存
#
# 说明：llm2asic 无需安装，Python 入口通过 PYTHONPATH=. 指向本仓库源码。

PYTHON     ?= python3
PYTHONPATH := .
MODEL      ?= examples/llama_tiny/model.yaml
OUT        ?= build_out
IVLOG      := $(shell command -v iverilog 2>/dev/null)
YOSYS      := $(shell command -v yosys 2>/dev/null)

# 综合后端配置（PDK 可配置）
BACKEND    ?= fpga
PDK        ?= sky130hd
LIBERTY    ?=

.PHONY: all test rtl synth clean

all: test rtl

# 端到端：解析 -> 量化 -> 黄金参考 -> RTL -> 仿真逐位比对（worst=0）
rtl:
	@echo "==> [rtl_backend] 生成 RTL 到 $(OUT)/rtl 并仿真验证"
	@PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m llm2asic build \
		--model $(MODEL) --out $(OUT)
	@echo "==> [rtl] 完成。RTL: $(OUT)/rtl/  | 报告: $(OUT)/test_report.json"

# 测试套件
test:
	@echo "==> [pytest] 运行测试"
	@PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest

# Yosys 综合（开源流程）：在 RTL 目录运行，产出门级网表到 $(OUT)/synth/
synth: rtl
	@if [ -z "$(YOSYS)" ]; then echo "ERROR: 未找到 yosys，请先安装"; exit 1; fi
	@mkdir -p $(OUT)/synth
	@echo "==> [yosys] 综合 $(OUT)/rtl -> $(OUT)/synth/netlist.v (backend=$(BACKEND) pdk=$(PDK))"
	@PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m llm2asic synth --model $(MODEL) \
		--out $(OUT) --backend $(BACKEND) --pdk $(PDK) \
		$(if $(LIBERTY),--liberty $(LIBERTY))
	@echo "==> [synth] 完成。网表: $(OUT)/synth/netlist.v  | 日志: $(OUT)/synth/synth.log"
	@if [ -f "$(OUT)/synth/util_report.txt" ]; then \
		echo "==> [synth] 资源占用:"; grep -E "Number of cells|Chip area" $(OUT)/synth/synth.log | tail -6; \
	fi

clean:
	rm -rf $(OUT)
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "==> 已清理构建产物与缓存"
