#!/usr/bin/env bash
# 黄区 -> 蓝区：跑一条命令，把结果压成一份"小到能手工粘贴"的摘要。
#
# 用法:
#   ./.agent/sync/collect.sh pytest tests/e2e/accuracy/minicpmo_4_5/ -x
#   ./.agent/sync/collect.sh bash some_bench.sh
#   ./.agent/sync/collect.sh -d                 # 不重跑，只用已有日志重新生成摘要
#   NPU_SYNC_TAIL=300 ./.agent/sync/collect.sh -d
#
# 产出（默认写在当前目录，可用 NPU_SYNC_LOG 指定别的路径）:
#   ./temp/npu-run.log          全量日志（留在黄区备查）
#   ./temp/npu-run.digest.txt   精简摘要（这个才是你要复制回蓝区的）
set -uo pipefail

LOG="${NPU_SYNC_LOG:-./temp/npu-run.log}"
DIGEST="${LOG%.log}.digest.txt"
TAIL_N="${NPU_SYNC_TAIL:-80}"

write_digest() {
  local cmd="$1" rc="$2"
  {
    echo "=== NPU RUN DIGEST ==="
    echo "commit : $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
    echo "cmd    : $cmd"
    echo "exit   : $rc"
    echo

    echo "--- pytest 结论 ---"
    grep -aE '^(FAILED|ERROR) |^=+ .*(passed|failed|error|skipped).* =+$' "$LOG" | tail -40
    echo

    echo "--- 异常 / 报错行 ---"
    grep -anE 'Traceback \(most recent|^[A-Za-z_.]*(Error|Exception):|RuntimeError|AssertionError|CUDA|ASCEND|npu' "$LOG" \
      | tail -50
    echo

    echo "--- 末尾 ${TAIL_N} 行 ---"
    tail -"$TAIL_N" "$LOG"
  } >"$DIGEST" 2>&1
}

report() {
  echo
  echo "摘要: $DIGEST  ($(wc -c <"$DIGEST") bytes / $(wc -l <"$DIGEST") 行)"
  echo "全量: $LOG      ($(wc -l <"$LOG") 行)"
  echo
  echo "把摘要打到屏幕上方便复制： cat $DIGEST"
}

# -d / --digest-only: 不重跑，只用已有日志重新生成摘要（NPU 测试重跑很贵）
if [[ "${1:-}" == "-d" || "${1:-}" == "--digest-only" ]]; then
  if [[ ! -f "$LOG" ]]; then
    echo "找不到日志 $LOG，先跑一次： ./.agent/sync/collect.sh <你的测试命令>" >&2
    exit 1
  fi
  echo "用已有日志重新生成摘要（TAIL=${TAIL_N}）: $LOG"
  write_digest "$(sed -n 's/^cmd    : //p' "$DIGEST" 2>/dev/null | head -1)" \
               "$(sed -n 's/^exit   : //p' "$DIGEST" 2>/dev/null | head -1)"
  report
  exit 0
fi

if [[ $# -eq 0 ]]; then
  echo "用法: ./.agent/sync/collect.sh <命令...>   或   ./.agent/sync/collect.sh -d" >&2
  exit 2
fi

echo "运行: $*"
# tee 而不是重定向：长测试要能看到实时进度，同时留全量日志。
"$@" 2>&1 | tee "$LOG"
RC="${PIPESTATUS[0]}"
echo "退出码: $RC"

write_digest "$*" "$RC"
report
exit "$RC"
