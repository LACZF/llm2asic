#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# OpenROAD-flow-scripts: 把 llm2asic 生成的 verilog (RTL) 转换为 GDS。
#
#  用法:
#    ./scripts/openroad_gds.sh \
#        --rtl <rtl_dir>                 # 含 *_accel.sv、gemv_*.sv、*.mem
#        [--top <顶层模块>]               # 默认自动从 *_accel.sv 推断
#        [--flow <OpenROAD-flow-scripts 根目录>]
#        [--platform sky130hd]
#        [--design <design 名>]          # 默认 = 顶层模块
#        [--out <结果输出目录>]           # 默认: <rtl_dir>/../openroad_gds
#        [--skip-drt 0|1]                # 1(默认): 跳过详细布线(DRT), 从全局布线
#                                        #   直接出 GDS, 可显著降低内存峰值;
#                                        #   0: 跑完整详细布线(大设计可能 OOM)
#
#  流程: 把 RTL+*.mem 归置到 flow 的 designs/src/<design>/ 下，
#        生成 config.mk 与 constraint.sdc，然后调用 OpenROAD flow 的
#        synth -> floorplan -> place -> cts -> route -> finish -> gds
#        得到 6_final.gds。
#
#  前置条件:
#    1) 已编译 OpenROAD-flow-scripts 并把工具装到 <flow>/tools/install/
#       (openroad、yosys、klayout)。本脚本检测不到时给出明确报错。
#    2) 平台 (Platform=sky130hd) 的 LEF/LIB/GDS 文件已随 flow 就绪。
#    3) 足够内存/时长。注意: llm2asic 生成的 RTL 是巨大组合数据通路
#       (宽乘法 + .mem ROM 展开), 门级综合(abc) 在低内存机器上会 OOM,
#       可能需要调小模型或提升内存。
# ---------------------------------------------------------------------------
set -euo pipefail

FLOW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORFS_ROOT="${ORFS_ROOT:-}"
PLATFORM="sky130hd"
DESIGN=""
TOP=""
RTL_DIR=""
OUT_DIR=""
SKIP_DRT=1

usage() { sed -n '3,28p' "$0"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rtl)      RTL_DIR="$2"; shift 2;;
    --top)      TOP="$2"; shift 2;;
    --flow)     ORFS_ROOT="$2"; shift 2;;
    --platform) PLATFORM="$2"; shift 2;;
    --design)   DESIGN="$2"; shift 2;;
    --out)      OUT_DIR="$2"; shift 2;;
    --skip-drt) SKIP_DRT="$2"; shift 2;;
    *) echo "未知参数: $1"; usage; exit 1;;
  esac
done

[[ -n "$RTL_DIR" && -d "$RTL_DIR" ]] || { echo "错误: --rtl 需要有效的 RTL 目录"; exit 1; }
case "$SKIP_DRT" in 0|1) ;; *) echo "错误: --skip-drt 只接受 0 或 1"; exit 1;; esac

# 推断顶层模块
if [[ -z "$TOP" ]]; then
  TOP="$(ls "$RTL_DIR" | sed -n 's/_accel\.sv$//p' | head -1 || true)"
fi
[[ -n "$TOP" ]] || { echo "错误: 未找到 *_accel.sv, 请用 --top 指定顶层模块"; exit 1; }
TOP="$(basename "$TOP")"; TOP="${TOP%_accel}"; TOP="${TOP}_accel"
DESIGN="${DESIGN:-${TOP%_accel}}"
OUT_DIR="${OUT_DIR:-$(cd "$RTL_DIR/.." && pwd)/openroad_gds}"

# 定位 flow (OpenROAD-flow-scripts)
if [[ -z "$ORFS_ROOT" ]]; then
  for cand in \
    "$FLOW_DIR/../eda/back-end/OpenROAD-flow-scripts" \
    "$HOME/OpenROAD-flow-scripts" \
    "/home/user/git/eda/back-end/OpenROAD-flow-scripts"; do
    if [[ -f "$cand/flow/Makefile" ]]; then ORFS_ROOT="$cand"; break; fi
  done
fi
[[ -n "$ORFS_ROOT" && -f "$ORFS_ROOT/flow/Makefile" ]] \
  || { echo "错误: 找不到 OpenROAD-flow-scripts, 用 --flow 指定"; exit 1; }
ORFS_ROOT="$(cd "$ORFS_ROOT" && pwd)"

# 检查已编译工具 (flow 实际读取 tools/install/...)
need=(
  "$ORFS_ROOT/tools/install/OpenROAD/bin/openroad"
  "$ORFS_ROOT/tools/install/yosys/bin/yosys"
)
for b in "${need[@]}"; do
  [[ -x "$b" ]] || { echo "错误: 缺少已编译工具 $b"; echo "请先在本机编译 OpenROAD-flow-scripts (build_openroad.sh -o) 并安装 KLayout"; exit 1; }
done
if [[ ! -d "$ORFS_ROOT/tools/install/klayout" ]]; then
  echo "警告: 未发现 tools/install/klayout, GDS merge/DRC 需 KLayout"
fi

echo "==> top=$TOP design=$DESIGN platform=$PLATFORM skip-drt=$SKIP_DRT"
echo "==> flow=$ORFS_ROOT  rtl=$RTL_DIR  out=$OUT_DIR"
if [[ "$SKIP_DRT" == 0 ]]; then
  echo "警告: --skip-drt 0 将执行完整详细布线, 大设计(数万门以上)可能超出 10GB 内存导致 OOM"
fi

FLOW="$ORFS_ROOT/flow"
SRC_V="$(cd "$FLOW/designs/src" && pwd)/$DESIGN"
mkdir -p "$SRC_V" "$OUT_DIR"
# 调整成绝对路径: ORFS `make -C flow` 以 flow/ 为 CWD, 相对 WORK_HOME 会
# 落到 flow/ 下而不是调用方的 --out 目录。
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

# 归置 verilog + .mem 到 flow designs/src/<design>/ (cp -u: 已存在则不触碰 mtime,
# 避免 make 因源文件变新而重新综合)
cp -fu "$RTL_DIR"/*.sv "$SRC_V"/ 2>/dev/null || true
cp -fu "$RTL_DIR"/*.mem "$SRC_V"/ 2>/dev/null || true
# yosys 以 flow/ 为 CWD 运行, $readmemh("xxx.mem") 相对 CWD 解析,
# 因此 mem 也复制到 flow 根目录, 避免 ROM 内容丢失(全部为 0)。
cp -fu "$RTL_DIR"/*.mem "$FLOW"/ 2>/dev/null || true

VFILES=""
for f in "$SRC_V"/*.sv; do
  b="$(basename "$f")"
  case "$b" in sim_tb.sv) continue;; esac
  VFILES="$VFILES $b"
done
VFILES="$(echo $VFILES)"
# VERILOG_FILES 用相对于 flow 的路径 (DESIGN_HOME/src/<nick>/*.sv 结构),
# 与示例 gcd 一致; DESIGN_NAME 必须是顶层模块名 (可不同于 nickname)。
VFABS=""
for b in $VFILES; do VFABS="$VFABS \$(DESIGN_HOME)/src/\$(DESIGN_NICKNAME)/$b"; done
VFABS="$(echo $VFABS)"

# 时钟约束: 默认 10ns; 只约束顶层 clk 输入
cat > "$SRC_V/$DESIGN.sdc" <<EOF
set CLK_PERIOD [expr {50.0}]
create_clock -period \$CLK_PERIOD -name clk [get_ports clk]
set_clock_uncertainty 0.1 [get_clocks clk]
# set_input_delay 0.2 -clock clk [remove_from_collection [all_inputs] [get_ports clk]]
# (本 OpenROAD 构建未注册 remove_from_collection, 改用简单写法)
set_input_delay 0.2 -clock clk [all_inputs]
set_output_delay 0.2 -clock clk [all_outputs]
EOF

# 生成设计 config.mk
DESIGN_CONFIG_DIR="$FLOW/designs/$PLATFORM/$DESIGN"
mkdir -p "$DESIGN_CONFIG_DIR"
cat > "$DESIGN_CONFIG_DIR/config.mk" <<EOF
export DESIGN_NAME     = $TOP
export DESIGN_NICKNAME = $DESIGN
export PLATFORM        = $PLATFORM

# llm2asic 生成的 RTL + mem (相对 flow/designs/src/<design>/, 与 gcd 示例一致)
export VERILOG_FILES = $VFABS
export SDC_FILE      = $SRC_V/$DESIGN.sdc

# 简化: 面积优先, 不追求时序
export CORE_UTILIZATION = 45
export TNS_END_PERCENT  = 100
export SYNTH_ADDER_TYPE = YOSYS
export PLACE_DENSITY    = 0.55
export GLOBAL_ROUTE_ARGS = -allow_congestion

# 无宏/macro 设计: 关闭 hierarchy 相关
export OPENROAD_HIERARCHICAL = 0

# llm2asic 的激活函数 LUT 是巨型 ROM (rsqrt/recip/exp/gelu), 会被当作电路内容
# 真实综合; 远超 ORFS 默认的 4096bit 限制, 故放开该上限。
export SYNTH_MEMORY_MAX_BITS = 20000000

# sky130hd 平台默认启用 adder 提取 (cells_adders_hd.v), 在 64 位宽加法树
# 上耗时极长; 仿照 gcd 示例禁用, 交 yosys 原生 techmap 映射。
export ADDER_MAP_FILE :=

# 详细布线开关: 1=跳过 DRT 从全局布线直接出 GDS(省内存); 0=完整详细布线。
export SKIP_DETAILED_ROUTE = $SKIP_DRT
EOF

# config.mk / sdc 每次都会被本脚本重写, 若保留"当前" mtime, ORFS make 会认为
# 配置比已产出的 results 新, 从而把整条 synth->P&R 链路重跑一遍。这里把它
# 打回历史时间(内容未变), 使流程幂等: 已有阶段自动跳过。
touch -d "2020-01-01 00:00:00" "$DESIGN_CONFIG_DIR/config.mk" "$SRC_V/$DESIGN.sdc"

echo "==> 生成 config: $DESIGN_CONFIG_DIR/config.mk"
echo "==> 运行 OpenROAD flow (synth...gds)，请耐心等待…"

run_step() { # $1 = target; 失败返回非零,不退出
  local t="$1"
  echo "---- make $t ----"
  make -C "$FLOW" DESIGN_CONFIG="$DESIGN_CONFIG_DIR/config.mk" \
       WORK_HOME="$OUT_DIR" "$t" 2>&1 | tee -a "$OUT_DIR/flow_steps.log"
  return ${PIPESTATUS[0]}
}

for t in synth floorplan place cts route finish; do
  if ! run_step "$t"; then
    # OpenSTA 的 Netlist parser 不接受 "output signed/wire signed" 声明
    # (STA-0171 语法错误); 剥离后重试一次。
    if [[ "$t" == synth ]]; then
      NET="$OUT_DIR/results/$PLATFORM/$DESIGN/base/1_2_yosys.v"
      echo "==> 剥离 $NET 中的 signed 声明后重试 synth"
      sed -ri 's/^(( *)(input|output|wire)) +signed /\1 /g' "$NET"
      run_step synth || { echo "错误: make synth 重试失败"; exit 1; }
    elif [[ "$t" == finish ]]; then
      # ORFS `finish` 末尾会跑 KLayout merge(6_1_merged.gds), 本机构建在
      # RUN_CMD 引号上有 bug 必失败; 该步骤由下方 def2stream 手工替代。
      echo "==> make finish 的 GDS-merge 步骤失败(已知 bug, 忽略), 交由下方 def2stream 生成 GDS"
    else
      echo "错误: make $t 失败"; exit 1
    fi
  fi
done

# ---------------------------------------------------------------------------
# 由 DEF 转 GDS (stream out)
# 说明: 本机 ORFS 构建的 `make gds`(KLayout merge)在 RUN_CMD 引号上有 bug,
#       且需显式传入平台标准单元的 GDS 作 in_files; 这里直接走 KLayout
#       def2stream.py, 与 `make gds` 的产物等价。
# ---------------------------------------------------------------------------
RES="$OUT_DIR/results/$PLATFORM/$DESIGN/base"
OBJ="$OUT_DIR/objects/$PLATFORM/$DESIGN/base"

# 1) 生成 klayout.lyt (make 目标按文件路径触发, 跳过有 bug 的 merge 封装)
run_step "$OBJ/klayout.lyt"

# 2) 收集平台标准单元 GDS (in_files), 让所有 LEF cell 都有对应几何
PGDS_DIR="$ORFS_ROOT/flow/platforms/$PLATFORM/gds"
IN_FILES=""
for g in "$PGDS_DIR"/*.gds; do
  [[ -f "$g" ]] && IN_FILES="${IN_FILES:+$IN_FILES }$g"
done
[[ -n "$IN_FILES" ]] || echo "警告: 平台目录未找到 *.gds, 生成出的 cell 几何为空"

# 3) DEF -> GDS
if [[ -x "$ORFS_ROOT/tools/install/klayout/klayout" ]]; then
  export KLAYOUT_CMD="$ORFS_ROOT/tools/install/klayout/klayout"
  bash "$FLOW/scripts/klayout.sh" -zz \
    -rd design_name="$TOP" \
    -rd in_def="$RES/6_final.def" \
    -rd in_files="$IN_FILES" \
    -rd seal_file="" \
    -rd out_file="$RES/6_final.gds" \
    -rd tech_file="$OBJ/klayout.lyt" \
    -rd layer_map="" \
    -r "$FLOW/util/def2stream.py" 2>&1 | tee -a "$OUT_DIR/flow_steps.log"
else
  echo "错误: 未找到 KLayout (tools/install/klayout), 无法生成 GDS"
  exit 1
fi

GDS="$RES/6_final.gds"
[[ -f "$GDS" ]] && echo "==> PASS: GDS -> $GDS" || { echo "错误: 未产出 6_final.gds"; exit 1; }
