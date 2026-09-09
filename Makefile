# LLM2ASIC Makefile
#
# 在当前目录直接执行：
#   make            # 先跑测试，再生成 RTL 并仿真验证（默认目标）
#   make test       # 仅运行 pytest 测试套件
#   make rtl        # 端到端：生成 RTL + 黄金参考 + 仿真逐位比对
#                   #   每个模型的所有产物按目录名归类到 $(OUT)/<模型名>/
#   make rtl-all    # 一次性转换 examples/ 下全部模型 (逐个生成 RTL 并仿真)
#   make synth      # Yosys 综合，产出最终门级网表 netlist.v（需 yosys）
#                   #   BACKEND=fpga       -> synth_xilinx（默认）
#                   #   BACKEND=asic PDK=sky130hd LIBERTY=xxx.lib -> 标准单元网表
#   make gds        # 一键：RTL -> OpenROAD flow -> 最终 GDS
#                   #   依赖以 make rtl 生成 RTL(源码模型见 MODEL/OUT)
#                   #   默认 MODEL=examples/gpt2_tiny/model.yaml OUT=build；
#                   #   其它模型如 llama_tiny 用
#                   #     make gds MODEL=examples/llama_tiny/model.yaml OUT=build_llama
#                   #   GDS_SKIP_DRT=0|1   1(默认)跳过详细布线省内存
#                   #   GDS_TOP=<顶层名>    默认从 $(OUT)/rtl/*_accel.sv 推断
#                   #   ORFS=<OpenROAD-flow-scripts 根目录> 可选覆盖自动探测
#                   #   SINGLE=1           额外产出合并的单文件 RTL (*_single.sv)
#   make clean      # 清理构建产物与缓存
#
# 说明：llm2asic 无需安装，Python 入口通过 PYTHONPATH=. 指向本仓库源码。

PYTHON     ?= python3
PYTHONPATH := .
MODEL      ?= examples/gpt2_tiny/model.yaml
OUT        ?= build
# 每个模型按目录名归类: 所有产物(rtl/报告/综合/GDS)输出到 $(OUT)/<模型名>/
NAME      := $(shell basename $(dir $(MODEL)))
MODEL_OUT := $(OUT)/$(NAME)
IVLOG      := $(shell command -v iverilog 2>/dev/null)
YOSYS      := $(shell command -v yosys 2>/dev/null)

# 综合后端配置（PDK 可配置）
BACKEND    ?= fpga
PDK        ?= sky130hd
LIBERTY    ?=

# GDS 一键流程配置
GDS_PLATFORM   ?= sky130hd
GDS_SKIP_DRT   ?= 1
GDS_TOP        ?=
SINGLE         ?= 0
RTL_SINGLE     := $(if $(filter 1,$(SINGLE)),--single-file)

.PHONY: help all test rtl rtl-all synth gds lint lint-all clean

help:
	@echo "LLM2ASIC 常用命令（也直接支持 make 子命令: test/rtl/synth/gds/clean）"
	@echo ""
	@echo "  make            # 默认: test + rtl"
	@echo "  make help       # 显示本帮助"
	@echo "  make test       # 运行 pytest 测试套件"
	@echo "  make rtl        # 端到端: 解析->量化->黄金参考->RTL->仿真逐位比对"
	@echo "                  #   产物输出到 \$(OUT)/<模型名>/  (按名称自动归类)"
	@echo "                  #   额外: SINGLE=1 合并产出单文件 RTL (*_single.sv)"
	@echo "  make rtl-all    # 一次转换 examples/ 下全部模型(声明式+外部格式)为 RTL"
	@echo "  make lint       # Verilator 检查 RTL, 报告到 \$(MODEL_OUT)/lint_report.txt"
	@echo "  make lint-all   # 对 examples/ 下全部模型逐个 lint"
	@echo "  make synth      # Yosys 综合出网表 (BACKEND=fpga|asic, PDK, LIBERTY=)"
	@echo "  make gds        # 一键 RTL->OpenROAD flow->最终 GDS (需 iverilog/yosys/OpenROAD)"
	@echo "  make clean      # 清理构建产物与缓存"
	@echo ""
	@echo "常用变量:"
	@echo "  MODEL=路径      模型 YAML (默认 examples/gpt2_tiny/model.yaml)"
	@echo "  OUT=目录        输出根目录 (默认 build; 各模型在其下按名分类)"
	@echo "  SINGLE=1        额外输出单文件 RTL"
	@echo "  GDS_PLATFORM=   GDS 目标 PDK (默认 sky130hd)"
	@echo "  GDS_SKIP_DRT=1  1=跳过详细布线省内存 (默认)"
	@echo "  GDS_TOP=名字    顶层模块名 (默认从 OUT/*/rtl/*_accel.sv 推断)"
	@echo "  ORFS=路径       OpenROAD-flow-scripts 根目录 (自动探测)"
	@echo "  BACKEND=        synth 后端: fpga(默认)|asic   PDK=/LIBERTY= (asic 用)"
	@echo ""
	@echo "示例:"
	@echo "  make gds"
	@echo "  make gds MODEL=examples/llama_tiny/model.yaml OUT=build_llama"
	@echo "  make rtl SINGLE=1"
	@echo "  make rtl-all OUT=build_out"

all: test rtl

# 端到端：解析 -> 量化 -> 黄金参考 -> RTL -> 仿真逐位比对（worst=0）
rtl:
	@echo "==> [rtl_backend] 生成 RTL 到 $(MODEL_OUT)/rtl 并仿真验证 (model=$(NAME))"
	@PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m llm2asic build \
		--model $(MODEL) --out $(MODEL_OUT) $(RTL_SINGLE)
	@echo "==> [rtl] 完成。RTL: $(MODEL_OUT)/rtl/  | 报告: $(MODEL_OUT)/test_report.json"

# 一次性转换 examples/ 下全部可综合模型 (每个输出到 $(OUT)/<模型名>/)
# 支持的来源(每目录取一个): model.yaml(+weights.npz) 声明式 >
#   model.safetensors > model.onnx > model.bin(+model.yaml) 外部格式。
rtl-all:
	@set -e; \
	fail=0; n=0; \
	for m in $$(for d in examples/*/; do \
	  d=$${d%/}; src=""; \
	  [ -f "$$d/model.yaml" ] && [ -f "$$d/weights.npz" ] && src="$$d/model.yaml"; \
	  [ -z "$$src" ] && [ -f "$$d/model.safetensors" ] && src="$$d/model.safetensors"; \
	  [ -z "$$src" ] && [ -f "$$d/model.onnx" ] && src="$$d/model.onnx"; \
	  [ -z "$$src" ] && [ -f "$$d/model.bin" ] && [ -f "$$d/model.yaml" ] && src="$$d/model.bin"; \
	  [ -n "$$src" ] && echo "$$src"; \
	done | sort); do \
	  n=$$((n+1)); \
	  echo ""; echo "=========== [rtl-all] $$m ==========="; \
	  $(MAKE) rtl MODEL="$$m" OUT="$(OUT)" || fail=1; \
	done; \
	echo ""; \
	if [ "$$fail" -eq 0 ]; then \
		echo "==> [rtl-all] 全部 $$n 个模型成功"; \
	else \
		echo "==> [rtl-all] 存在失败模型 (见上方输出)"; exit 1; \
	fi

# Verilator lint：检查当前模型生成的 RTL, 报告写到 $(MODEL_OUT)/lint_report.txt
lint: rtl
	@bash scripts/lint_rtl.sh --rtl $(MODEL_OUT)/rtl --report $(MODEL_OUT)/lint_report.txt

# 对 examples/ 全部模型逐个 lint（来源发现同 rtl-all）
lint-all:
	@set -e; \
	fail=0; n=0; \
	for m in $$(for d in examples/*/; do \
	  d=$${d%/}; src=""; \
	  [ -f "$$d/model.yaml" ] && [ -f "$$d/weights.npz" ] && src="$$d/model.yaml"; \
	  [ -z "$$src" ] && [ -f "$$d/model.safetensors" ] && src="$$d/model.safetensors"; \
	  [ -z "$$src" ] && [ -f "$$d/model.onnx" ] && src="$$d/model.onnx"; \
	  [ -z "$$src" ] && [ -f "$$d/model.bin" ] && [ -f "$$d/model.yaml" ] && src="$$d/model.bin"; \
	  [ -n "$$src" ] && echo "$$src"; \
	done | sort); do \
	  n=$$((n+1)); \
	  echo ""; echo "=========== [lint-all] $$m ==========="; \
	  $(MAKE) lint MODEL="$$m" OUT="$(OUT)" || fail=1; \
	done; \
	echo ""; \
	if [ "$$fail" -eq 0 ]; then \
		echo "==> [lint-all] 全部 $$n 个模型 LINT 通过"; \
	else \
		echo "==> [lint-all] 存在失败模型 (见上方输出)"; exit 1; \
	fi

# 测试套件
test:
	@echo "==> [pytest] 运行测试"
	@PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest

# Yosys 综合（开源流程）：在 RTL 目录运行，产出门级网表到 $(OUT)/synth/
synth: rtl
	@if [ -z "$(YOSYS)" ]; then echo "ERROR: 未找到 yosys，请先安装"; exit 1; fi
	@mkdir -p $(MODEL_OUT)/synth
	@echo "==> [yosys] 综合 $(MODEL_OUT)/rtl -> $(MODEL_OUT)/synth/netlist.v (backend=$(BACKEND) pdk=$(PDK))"
	@PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m llm2asic synth --model $(MODEL) \
		--out $(MODEL_OUT) --backend $(BACKEND) --pdk $(PDK) \
		$(if $(LIBERTY),--liberty $(LIBERTY))
	@echo "==> [synth] 完成。网表: $(MODEL_OUT)/synth/netlist.v  | 日志: $(MODEL_OUT)/synth/synth.log"
	@if [ -f "$(MODEL_OUT)/synth/util_report.txt" ]; then \
		echo "==> [synth] 资源占用:"; grep -E "Number of cells|Chip area" $(MODEL_OUT)/synth/synth.log | tail -6; \
	fi

# GDS 一键流程：RTL(依赖 rtl 目标) -> OpenROAD flow -> 最终 GDS
#   RTL 由 `make rtl` 生成(模型来自 MODEL, 输出到 OUT/rtl), 这里只做
#   归置 -> 综合 -> 布局布线 -> GDS StreamOut。默认走 gpt2_tiny(在
#   ACT=16 下可逐位通过; llama_tiny 需 ACT=24, 见 numeric.py)。示例:
#     make gds
gds: rtl
	set -e; \
	NICK="$(GDS_TOP)"; [ -n "$$NICK" ] || NICK="$$(ls $(MODEL_OUT)/rtl | sed -n 's/_accel\.sv$$//p' | head -1)"; \
	[ -n "$$NICK" ] || { echo "错误: 未从 $(MODEL_OUT)/rtl 推断出顶层模块, 请用 GDS_TOP= 指定"; exit 1; }; \
	echo "==> [1/2] 运行 OpenROAD flow (synth floorplan place cts route finish + GDS StreamOut)"; \
	bash scripts/openroad_gds.sh \
		--rtl $(MODEL_OUT)/rtl \
		--out $(MODEL_OUT)/openroad_gds \
		--platform $(GDS_PLATFORM) \
		--skip-drt $(GDS_SKIP_DRT) \
		$(if $(GDS_TOP),--top $(GDS_TOP)) \
		$(if $(ORFS),--flow $(ORFS)); \
	echo "==> [2/2] 复制最终 GDS 到固定路径"; \
	cp -f $(MODEL_OUT)/openroad_gds/results/$(GDS_PLATFORM)/$$NICK/base/6_final.gds $(OUT)/$$NICK.gds; \
	echo "==> PASS: GDS -> $(OUT)/$$NICK.gds"

clean:
	rm -rf $(OUT)
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "==> 已清理构建产物与缓存"
