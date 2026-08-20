#!/usr/bin/env bash
# 蓝区 -> 黄区：把当前工作树（含未提交、未跟踪的改动）推到一个专用 scratch 分支。
#
# 不会碰你的本地 index 和 working tree，也不会在你的功能分支上留下 WIP 提交：
# 用一个临时 index 把工作树打包成一个游离 commit，直接 push 上去。
#
# 用法:
#   ./.agent/sync/push-to-npu.sh
#   NPU_SYNC_BRANCH=wip/other ./.agent/sync/push-to-npu.sh
set -euo pipefail

BRANCH="${NPU_SYNC_BRANCH:-wip/npu-main}"
REMOTE="${NPU_SYNC_REMOTE:-origin}"

cd "$(git rev-parse --show-toplevel)"

# 上次推上去的版本，用来算"黄区这次会收到什么"。以远端为准，避免本地跟踪 ref 过期。
PREV="$(git ls-remote "$REMOTE" "refs/heads/$BRANCH" 2>/dev/null | cut -f1)"

TMP_INDEX="$(mktemp -t npu-sync-index.XXXXXX)"
trap 'rm -f "$TMP_INDEX"' EXIT

export GIT_INDEX_FILE="$TMP_INDEX"
git read-tree HEAD
git add -A .
TREE="$(git write-tree)"
unset GIT_INDEX_FILE

# 黄区完全靠这些脚本干活。要是被忽略规则挡掉（历史上出过：一条裸 .agent/ 盖掉了
# !/.agent/sync/ 的放行），黄区就拿不到 pull/collect，只能手打长命令 —— 直接拦住。
MISSING=()
for f in pull-on-npu.sh collect.sh; do
  git ls-tree -r --name-only "$TREE" -- ".agent/sync/$f" | grep -q . || MISSING+=(".agent/sync/$f")
done
if (( ${#MISSING[@]} )); then
  echo "错误：以下 sync 脚本没进这次推送，黄区会拿不到：" >&2
  printf '  %s\n' "${MISSING[@]}" >&2
  echo >&2
  echo "多半是被忽略规则挡了，查一下：" >&2
  printf '  git check-ignore -v %s\n' "${MISSING[@]}" >&2
  echo "修 .git/info/exclude 里的 /.agent/ 那几条（见 .agent/sync/README.md 第五节）。" >&2
  exit 1
fi

BASE="$(git rev-parse HEAD)"
SRC_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
COMMIT="$(git commit-tree "$TREE" -p "$BASE" \
  -m "npu-sync: ${SRC_BRANCH}@$(git rev-parse --short HEAD) $(date -u +%Y-%m-%dT%H:%M:%SZ)")"

git push -q --force "$REMOTE" "$COMMIT:refs/heads/$BRANCH"

echo "已推送 ${COMMIT:0:12} -> ${REMOTE}/${BRANCH}"
echo "  基于  ${BASE:0:12}  (${SRC_BRANCH})"

UNCOMMITTED="$(git diff --name-only "$BASE" "$COMMIT" | wc -l)"
echo "  其中未提交改动 ${UNCOMMITTED} 个文件"

echo
if [[ -z "$PREV" ]]; then
  echo "首次推送该分支，黄区将收到完整代码库。"
elif ! git cat-file -e "$PREV^{commit}" 2>/dev/null; then
  echo "上次同步的 commit ${PREV:0:12} 不在本地，无法比对变化。"
  echo "（先 git fetch $REMOTE $BRANCH 再推可看到差异）"
elif [[ "$(git rev-parse "$PREV^{tree}")" == "$TREE" ]]; then
  # 比 tree 而不是 commit sha —— commit message 带时间戳，sha 必然不同
  echo "内容与上次推送完全相同，黄区无需重新同步。"
else
  # 这才是黄区真正会收到的变化：相对上次同步的差异
  echo "黄区将收到的变化（相对上次同步 ${PREV:0:12}）："
  git --no-pager diff --stat "$PREV" "$COMMIT" | tail -30
fi

echo
echo "黄区执行： ./.agent/sync/pull-on-npu.sh"
