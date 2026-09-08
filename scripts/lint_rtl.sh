#!/usr/bin/env bash
# 对生成的 RTL 运行 Verilator lint, 产出报告文件。
#
# 用法:
#   lint_rtl.sh --rtl <rtl_dir> --report <report.txt> [--top <顶层模块>]
#
# 顶层未指定时自动取 <rtl_dir>/*_accel.sv。
# 只 lint 当前设计实际用到的文件(顶层 + 顶层实例化的模块), 排除
# sim_tb / *_single.sv 以及目录里其他模型的残留文件。
#
# 判定: 存在真实 %Error 或"未在豁免清单内"的 Warning 类别 => FAIL。
# 豁免类别(LINT_WNO)对应本项目生成式 RTL 的固有风格: 定宽总线截断/
# 扩展、未用参数、顺传参数、顺序块内阻塞赋值(计算临时量)等。

set -uo pipefail

RTL_DIR=""
REPORT=""
TOP=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rtl)      RTL_DIR="$2"; shift 2;;
    --report)   REPORT="$2"; shift 2;;
    --top)      TOP="$2"; shift 2;;
    *) echo "未知参数: $1"; exit 2;;
  esac
done

[[ -n "$RTL_DIR" && -d "$RTL_DIR" ]] || { echo "错误: --rtl 需要有效的 RTL 目录"; exit 2; }
[[ -n "$REPORT" ]] || { echo "错误: --report 需要报告文件路径"; exit 2; }

if ! command -v verilator >/dev/null 2>&1; then
  cat > "$REPORT" <<EOF
[lint] verilator 未安装, 请先安装 Verilator (apt install verilator)
EOF
  cat "$REPORT"
  exit 1
fi

if [[ -z "$TOP" ]]; then
  TOP="$(ls "$RTL_DIR"/*_accel.sv 2>/dev/null | head -1 || true)"
  [[ -n "$TOP" ]] || { echo "错误: 未在 $RTL_DIR 找到 *_accel.sv"; exit 2; }
  TOP="$(basename "$TOP")"; TOP="${TOP%.sv}"
fi

# 顶层实例化的子模块(模块名=文件名): gemv_N / rmsnorm / layernorm / attn
subs=$(grep -oE '^\s*[A-Za-z_][A-Za-z0-9_]*\s*#' "$RTL_DIR/$TOP.sv" \
       | awk '{print $1}' | sort -u)
FILES="$RTL_DIR/$TOP.sv"
for s in $subs; do
  if [[ -f "$RTL_DIR/$s.sv" ]]; then FILES="$FILES $RTL_DIR/$s.sv"; fi
done

# 生成式代码的固有告警类别(无害, 报告里照常列数字):
#   WIDTHTRUNC/WIDTHEXPAND/WIDTHCONCAT 固定点截断/扩展
#   UNUSEDSIGNAL/UNUSEDPARAM/UNUSED    模板参数与临时量
#   GENUNNAMED/SETUPVAL/BLKSEQ/UNOPTFLAT 生成式风格
#   DECLFILENAME/PINMISSING/PINNOTFOUND/CONSTRAINTIGN/UNPACKED 等
WNO_DEFAULT="WIDTHTRUNC WIDTHEXPAND WIDTHCONCAT UNUSEDSIGNAL UNUSEDPARAM \
UNUSED GENUNNAMED BLKSEQ UNOPTFLAT DECLFILENAME PINMISSING PINNOTFOUND \
CONSTRAINTIGN TIMESCALEMOD VARHIDDEN STMTDLY ASSIGNDLY"
WNO="${LINT_WNO:-$WNO_DEFAULT}"
WNO_FLAGS=""
for c in $WNO; do WNO_FLAGS="$WNO_FLAGS -Wno-$c"; done

VDIR="$(mktemp -d)/verilint"
LOG="$REPORT.raw"
if ! verilator --lint-only -Wall --top-module "$TOP" -Mdir "$VDIR" $WNO_FLAGS \
     $FILES > "$LOG" 2>&1; then
  rc=$?
  # verilator 对剩余告警也会退出非 0; 以 log 内容判定真正错误/新告警
else
  rc=0
fi

# 分类统计
errors=0; warns=0
declare -A WCNT
while IFS= read -r line; do
  if [[ "$line" =~ %Error-([A-Z0-9]+) ]]; then errors=$((errors+1)); continue; fi
  if [[ "$line" =~ %Warning-([A-Z0-9]+) ]]; then
    cat_name="${BASH_REMATCH[1]}"
    warns=$((warns+1))
    WCNT["$cat_name"]=$(( ${WCNT["$cat_name"]:-0} + 1 ))
    continue
  fi
done < "$LOG"

# 非豁免告警类别(说明清单之外 = 值得检查)
unexpected=()
for k in "${!WCNT[@]}"; do
  [[ " $WNO " == *" $k "* ]] || unexpected+=("$k(${WCNT[$k]})")
done

ts="$(date '+%Y-%m-%d %H:%M:%S')"
vver="$(verilator --version 2>/dev/null | head -1)"
{
  echo "================================================================"
  echo " RTL Lint 报告              $ts"
  echo " 顶层模块 : $TOP"
  echo " lint文件 : $(echo $FILES | wc -w) 个: $(echo $FILES | tr ' ' '\n' | sed 's#.*/##' | tr '\n' ' ')"
  echo " 工具版本 : $vver"
  echo " 豁免类别 : $WNO"
  echo "---------------------------------------------------------------"
  echo " 尚未豁免的 Warning 类别与计数:"
  if [[ ${#WCNT[@]} -eq 0 ]]; then
    echo "   (无)"
  else
    for k in "${!WCNT[@]}"; do echo "   $k : ${WCNT[$k]}"; done | sort
  fi
  echo " Error 数 : $errors"
  echo "---------------------------------------------------------------"
} > "$REPORT"
cat "$LOG" >> "$REPORT"
if [[ $errors -gt 0 ]]; then
  verdict="FAIL: $errors 个 Error"
elif [[ ${#unexpected[@]} -gt 0 ]]; then
  verdict="FAIL: 出现未免疫的告警类别 ${unexpected[*]}"
else
  verdict="PASS: 无未豁免告警($warns 条已豁免类别中的告警已记录)"
fi
echo "================================================================" >> "$REPORT"
echo " verdict : $verdict" >> "$REPORT"

echo "==> [lint] $TOP: errors=$errors warns=$warns unexpected=${#unexpected[@]}"
echo "==> [lint] $verdict"
[[ $errors -eq 0 && ${#unexpected[@]} -eq 0 ]]