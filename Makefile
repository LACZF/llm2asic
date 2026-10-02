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
#   make hls        # GraphIR -> HLS C++(/ONNX) -> Bambu -> Verilog（独立于 rtl）
#                   #   HLS_BACKEND=native(默认)|hls4ml|onnx
#                   #   自动探测 bambu: 没装则停在 C++/ONNX 并提示
#                   #   HLS_STRICT=1  缺 bambu 时直接失败
#                   #   NO_YOSYS_CHECK=1 跳过慢的 Yosys 展开检查(~10min)
#   make hls-all    # 一次跑完 native/hls4ml/onnx 三条 HLS 路径
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

# HLS 后端配置（与 rtl/synth 流程并列，互不影响）
HLS_BACKEND ?= native
HLS_DEVICE  ?=
HLS_CLOCK_NS ?=
HLS_COMPILER ?=
HLS_EVAL    ?=
# Bambu 自动探测：没装就自动停在 C++/ONNX（并打印提示），
# 装了则默认一路走到 Verilog。HLS_STRICT=1 可要求缺 Bambu 时直接失败。
BAMBU       := $(shell command -v bambu 2>/dev/null)
HLS_STRICT  ?= 0
NO_BAMBU    ?= $(if $(strip $(BAMBU)),0,1)
HLS_NO_BAMBU_FLAG := $(if $(filter 1,$(NO_BAMBU)),--no-bambu)
# Yosys 展开检查对 gpt2_tiny 要 ~10 分钟(1241 个模块)，NO_YOSYS_CHECK=1 可跳过
NO_YOSYS_CHECK ?= 0
HLS_NO_YOSYS_FLAG := $(if $(filter 1,$(NO_YOSYS_CHECK)),--no-yosys-check)
ifeq ($(strip $(BAMBU)),)
HLS_BAMBU_MISSING := 1
# 每条单独一行, 且不得含单/双引号或反引号(会被 shell 打印)
HLS_NOTE1 := ==> NOTE: 未检测到 bambu 可执行文件，本次只生成 HLS C++/ONNX，不综合成 Verilog。
HLS_NOTE2 := ==>       安装 Bambu 后重跑即可得到 Verilog: https://github.com/ferrandi/PandA-bambu
HLS_NOTE3 := ==>       强制要求 bambu 时用: make hls HLS_STRICT=1
endif

# GDS 一键流程配置
GDS_PLATFORM   ?= sky130hd
GDS_SKIP_DRT   ?= 1
GDS_TOP        ?=
SINGLE         ?= 0
RTL_SINGLE     := $(if $(filter 1,$(SINGLE)),--single-file)

.PHONY: help all test rtl rtl-all synth gds lint lint-all hls hls-all clean

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
	@echo "  make hls        # HLS 路径: GraphIR -> C++(/ONNX) -> Bambu -> Verilog"
	@echo "                  #   自动探测 bambu; 缺 Bambu 则停在 C++/ONNX 并提示"
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
	@echo "  HLS_BACKEND=    HLS 路径: native(默认)|hls4ml|onnx"
	@echo "  HLS_STRICT=1    缺 bambu 时直接失败(默认自动降级为只出 C++)"
	@echo "  HLS_DEVICE=/HLS_CLOCK_NS=/HLS_COMPILER=/HLS_EVAL=  透传给 Bambu"
	@echo ""
	@echo "示例:"
	@echo "  make gds"
	@echo "  make gds MODEL=examples/llama_tiny/model.yaml OUT=build_llama"
	@echo "  make rtl SINGLE=1"
	@echo "  make rtl-all OUT=build_out"
	@echo "  make hls NO_BAMBU=1"
	@echo "  make hls HLS_BACKEND=hls4ml NO_BAMBU=1 OUT=build_hls"

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

# HLS 路径（与 rtl 并列，独立开关）：GraphIR -> C++(/ONNX) -> Bambu -> Verilog
#   HLS_BACKEND=native   自研 C 内核（默认，无需第三方 HLS 工具）
#   HLS_BACKEND=hls4ml   走 hls4ml 生成的 HLS C++ 工程
#   HLS_BACKEND=onnx     只导出单步 decode ONNX
#   NO_BAMBU=1           停在 C++/ONNX，不调用 Bambu（无 HLS 工具链时用）
hls:
	$(if $(HLS_BAMBU_MISSING),@echo '$(HLS_NOTE1)')
	$(if $(HLS_BAMBU_MISSING),@echo '$(HLS_NOTE2)')
	$(if $(HLS_BAMBU_MISSING),@echo '$(HLS_NOTE3)')
	@if [ -n "$(HLS_BAMBU_MISSING)" ] && [ "$(HLS_STRICT)" = "1" ]; then \
		echo "==> ERROR: HLS_STRICT=1 但未找到 bambu"; exit 1; fi
	@echo "==> [hls_backend] $(HLS_BACKEND) -> $(MODEL_OUT)/hls (bambu=$(if $(filter 1,$(NO_BAMBU)),关,开))"
	@PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m llm2asic hls \
		--model $(MODEL) --out $(MODEL_OUT) \
		--hls-backend $(HLS_BACKEND) \
		$(HLS_NO_BAMBU_FLAG) \
		$(if $(HLS_DEVICE),--device $(HLS_DEVICE)) \
		$(if $(HLS_CLOCK_NS),--clock-period $(HLS_CLOCK_NS)) \
		$(if $(HLS_COMPILER),--compiler $(HLS_COMPILER)) \
		$(HLS_NO_YOSYS_FLAG) \
		$(if $(HLS_EVAL),--evaluate $(HLS_EVAL))
	@echo "==> [hls] 完成。产物: $(MODEL_OUT)/hls/"

# 三条 HLS 路径各跑一遍，输出到 $(MODEL_OUT)/hls/<backend>/
hls-all:
	@set -e; for b in native hls4ml onnx; do \
		echo "==> [hls_backend] $$b"; \
		$(MAKE) --no-print-directory hls HLS_BACKEND=$$b \
			NO_BAMBU=$(NO_BAMBU) HLS_STRICT=$(HLS_STRICT) \
			NO_YOSYS_CHECK=$(NO_YOSYS_CHECK) \
			HLS_DEVICE=$(HLS_DEVICE) \
			HLS_CLOCK_NS=$(HLS_CLOCK_NS) HLS_COMPILER=$(HLS_COMPILER) \
			HLS_EVAL=$(HLS_EVAL) OUT=$(OUT) MODEL=$(MODEL); \
	done
	@echo "==> [hls-all] 三条路径完成:"
	@echo "     native : $(MODEL_OUT)/hls/hls/llm2asic_kernel.cpp"
	@echo "     hls4ml : $(MODEL_OUT)/hls/hls4ml/<project>/<project>_bridge.cpp"
	@echo "     onnx   : $(MODEL_OUT)/hls/*.onnx"

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
