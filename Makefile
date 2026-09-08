# LLM2ASIC Makefile
#
# 在当前目录直接执行：
#   make            # 先跑测试，再生成 RTL 并仿真验证（默认目标）
#   make test       # 仅运行 pytest 测试套件
#   make rtl        # 端到端：生成 RTL + 黄金参考 + 仿真逐位比对
#   make synth      # Yosys 综合，产出最终门级网表 netlist.v（需 yosys）
#                   #   BACKEND=fpga       -> synth_xilinx（默认）
#                   #   BACKEND=asic PDK=sky130hd LIBERTY=xxx.lib -> 标准单元网表
#   make gds        # 一键：RTL -> OpenROAD flow -> 最终 GDS
#                   #   依赖以 make rtl 生成 RTL(源码模型见 MODEL/OUT)
#                   #   默认 MODEL=examples/gpt2_tiny/model.yaml OUT=build_gds；
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
OUT        ?= build_gds
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

.PHONY: all test rtl synth gds clean

all: test rtl

# 端到端：解析 -> 量化 -> 黄金参考 -> RTL -> 仿真逐位比对（worst=0）
rtl:
	@echo "==> [rtl_backend] 生成 RTL 到 $(OUT)/rtl 并仿真验证"
	@PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m llm2asic build \
		--model $(MODEL) --out $(OUT) $(RTL_SINGLE)
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

# GDS 一键流程：RTL(依赖 rtl 目标) -> OpenROAD flow -> 最终 GDS
#   RTL 由 `make rtl` 生成(模型来自 MODEL, 输出到 OUT/rtl), 这里只做
#   归置 -> 综合 -> 布局布线 -> GDS StreamOut。默认走 gpt2_tiny(在
#   ACT=16 下可逐位通过; llama_tiny 需 ACT=24, 见 numeric.py)。示例:
#     make gds
gds: rtl
	set -e; \
	NICK="$(GDS_TOP)"; [ -n "$$NICK" ] || NICK="$$(ls $(OUT)/rtl | sed -n 's/_accel\.sv$$//p' | head -1)"; \
	[ -n "$$NICK" ] || { echo "错误: 未从 $(OUT)/rtl 推断出顶层模块, 请用 GDS_TOP= 指定"; exit 1; }; \
	echo "==> [1/2] 运行 OpenROAD flow (synth floorplan place cts route finish + GDS StreamOut)"; \
	bash scripts/openroad_gds.sh \
		--rtl $(OUT)/rtl \
		--out $(OUT)/openroad_gds \
		--platform $(GDS_PLATFORM) \
		--skip-drt $(GDS_SKIP_DRT) \
		$(if $(GDS_TOP),--top $(GDS_TOP)) \
		$(if $(ORFS),--flow $(ORFS)); \
	echo "==> [2/2] 复制最终 GDS 到固定路径"; \
	cp -f $(OUT)/openroad_gds/results/$(GDS_PLATFORM)/$$NICK/base/6_final.gds $(OUT)/$$NICK.gds; \
	echo "==> PASS: GDS -> $(OUT)/$$NICK.gds"

clean:
	rm -rf $(OUT)
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "==> 已清理构建产物与缓存"
