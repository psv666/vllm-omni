#!/usr/bin/env bash
# 黄区 NPU 机器：把仓库硬同步到蓝区刚推上来的版本。
#
# - fetch 走写死的完整 URL，不依赖本地 origin 指向哪里（黄区 origin 通常是 upstream）
# - 落到专用本地分支 npu-test，不会改写 main 或任何跟踪分支
# - 匿名 https，不需要 token（蓝区 fork 是 public）
#
# 用法:
#   ./.agent/sync/pull-on-npu.sh
#   ./.agent/sync/pull-on-npu.sh -f      # 确认丢弃黄区本地改动
set -euo pipefail

BRANCH="${NPU_SYNC_BRANCH:-wip/npu-main}"
URL="${NPU_SYNC_URL:-https://github.com/psv666/vllm-omni.git}"
LOCAL_BRANCH="${NPU_SYNC_LOCAL_BRANCH:-npu-main}"

cd "$(git rev-parse --show-toplevel)"

git fetch --force "$URL" "$BRANCH"

echo "蓝区版本："
git --no-pager log --oneline -1 FETCH_HEAD

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  if [[ "${1:-}" != "-f" ]]; then
    echo
    echo "黄区有未提交改动，同步会丢弃它们："
    git --no-pager status --short --untracked-files=no
    echo
    echo "确认丢弃 -> 加 -f 重跑；想留着 -> 先 git stash 或 git diff > /tmp/npu.patch"
    exit 1
  fi
fi

# -B 创建或重置专用分支，-f 丢弃本地改动；当前在 main 上也不会污染 main
git checkout -f -B "$LOCAL_BRANCH" FETCH_HEAD

# sync 脚本只在 npu-test 上是被跟踪的。切回 main 时 git 会把 .agent/sync/ 删掉，
# 下次就没有 pull 脚本可跑（要重新手打 bootstrap 长命令）。留一份在分支外面。
BOOTSTRAP="${NPU_SYNC_BOOTSTRAP:-$HOME/.npu-sync}"
if mkdir -p "$BOOTSTRAP" 2>/dev/null; then
  cp -f .agent/sync/pull-on-npu.sh .agent/sync/collect.sh "$BOOTSTRAP/" 2>/dev/null || true
fi

echo "同步完成：分支 ${LOCAL_BRANCH} @ $(git rev-parse --short HEAD)"
echo "  脚本副本：${BOOTSTRAP}/  (切到别的分支后可用它恢复)"
