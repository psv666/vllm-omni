# 蓝区 ⇄ 黄区 代码同步

在网络隔离的 NPU 服务器上测试蓝区改好的代码。改动经 GitHub 上一个专用 scratch 分支单向推过去，测试结果手工粘回。

```
  蓝区 (能用 AI，不能跑 NPU)                黄区 NPU 机器 (网络隔离，无 AI)
  /data/pengsiv/vllm-omni                   /data/pengsiv/vllm-omni
         │                                          │
         │  push-to-npu.sh                          │
         └──────► github.com/psv666/vllm-omni ──────┤  pull-on-npu.sh
                  分支 wip/npu-test                  │
                                                     │  collect.sh <测试命令>
         ◄─────── 手工复制 digest 文本 ──────────────┘
```

---

## 一、首次设置

### 蓝区（这台）

已经配好，无需操作。检查一下：

```bash
ls .agent/sync/          # 应有 push-to-npu.sh / pull-on-npu.sh / collect.sh / README.md
git remote get-url origin | sed 's#//[^@]*@#//***@#'   # 应指向 psv666/vllm-omni
```

### 黄区（NPU 机器）

在仓库根目录跑一次 bootstrap。此时脚本还没同步过去，所以要手打完整命令：

```bash
git fetch --force https://github.com/psv666/vllm-omni.git wip/npu-test \
  && git checkout -f -B npu-test FETCH_HEAD
```

> ⚠️ 用 `checkout -f -B`，**不要**用 `git reset --hard FETCH_HEAD`。后者会把你当前所在的分支（通常是 `main`）改写成 scratch commit，之后 log 和 pull 都会很乱。

建议顺手堵掉误 push（黄区 `origin` 指向官方仓库 vllm-project/vllm-omni）：

```bash
git remote set-url --push origin no-push
```

这样手滑 `git push` 会立刻报错，fetch 不受影响。

bootstrap 完成后，`.agent/sync/` 下的脚本就在黄区了，之后不用再手打长命令。

---

## 二、日常流程

### 1. 蓝区：推送

```bash
./.agent/sync/push-to-npu.sh
```

把**当前工作树**（含未提交、未跟踪的新文件）打包推到 `wip/npu-test`。

输出示例：

```
已推送 4295288da804 -> origin/wip/npu-test
  基于  e8c667d7d2c4  (minicpm_challenge_acc_perf_ci)
  其中未提交改动 4 个文件

黄区将收到的变化（相对上次同步 41ffb28f0a96）：
 .agent/sync/push-to-npu.sh | 24 +++++++++++++++++++++++-
 1 file changed, 23 insertions(+), 1 deletion(-)

黄区执行： ./.agent/sync/pull-on-npu.sh
```

**怎么读这份输出** —— 两个数字含义不同，别混：

| 行 | 含义 |
|---|---|
| `基于 <sha>` | scratch commit 的父提交，即你当前 HEAD。**HEAD 里已有的东西全都会带过去** |
| `其中未提交改动 N 个文件` | 工作树相对 HEAD 多出来的部分，仅供你确认没漏东西 |
| `黄区将收到的变化` | **相对上次同步的差异，这才是黄区实际会更新的内容** |

> 常见困惑：`git pull` 拉了 15 个文件后再 push，"未提交改动"可能只显示 1、2 个。这是对的 —— 那 15 个文件已经进了 HEAD，不算"未提交"，但它们照样在 scratch commit 里。看第三行才准。

**它不会碰你的本地状态**：不改 index、不改 working tree、不在你的功能分支上留 WIP 提交。推完 `git status` 和推之前一模一样。

### 2. 黄区：同步

```bash
./.agent/sync/pull-on-npu.sh
```

固定落在本地 `npu-test` 分支，你的 `main` 和其他分支不受影响。

有未提交改动时会先拦住你：

```
黄区有未提交改动，同步会丢弃它们：
 M README.md

确认丢弃 -> 加 -f 重跑；想留着 -> 先 git stash 或 git diff > /tmp/npu.patch
```

确认要丢就 `./.agent/sync/pull-on-npu.sh -f`。

### 3. 黄区：跑测试并收集结果

```bash
./.agent/sync/collect.sh pytest tests/e2e/accuracy/minicpmo_4_5/ -x
```

任何命令都行，不限于 pytest：

```bash
./.agent/sync/collect.sh bash tests/e2e/perf/run_bench.sh
./.agent/sync/collect.sh python -m vllm_omni.entrypoints.cli serve ...
```

产出两个文件：

| 文件 | 用途 |
|---|---|
| `/tmp/npu-run.log` | 全量日志，留在黄区备查 |
| `/tmp/npu-run.digest.txt` | **精简摘要，这个才是你要复制回蓝区的** |

摘要结构：

```
=== NPU RUN DIGEST ===
commit : bb61ef06                  ← 对应哪次同步，蓝区能对上
cmd    : pytest tests/... -x
exit   : 1

--- pytest 结论 ---              ← FAILED/ERROR 行 + 统计行
--- 异常 / 报错行 ---            ← Traceback / *Error / CUDA / ASCEND / npu 相关
--- 末尾 80 行 ---
```

脚本会打印字节数和行数，方便你判断粘贴量。

### 4. 复制回蓝区

```bash
cat /tmp/npu-run.digest.txt
```

全选复制，粘给 AI。摘要里带了 `commit`，蓝区能确认你测的是哪一版代码。

如果摘要不够用（比如要看更前面的日志），**不用重跑测试** —— 用 `-d` 拿已有日志重新生成摘要：

```bash
NPU_SYNC_TAIL=300 ./.agent/sync/collect.sh -d
```

或者直接从全量日志里取你要的那段：

```bash
sed -n '1200,1400p' /tmp/npu-run.log
grep -n "OutOfMemory" -A20 /tmp/npu-run.log
```

---

## 三、环境变量

三个脚本都可以用环境变量改行为，不用改代码。

| 变量 | 默认值 | 作用 | 用在 |
|---|---|---|---|
| `NPU_SYNC_BRANCH` | `wip/npu-test` | GitHub 上的 scratch 分支名 | push / pull |
| `NPU_SYNC_REMOTE` | `origin` | 蓝区推送用的 remote | push |
| `NPU_SYNC_URL` | `https://github.com/psv666/vllm-omni.git` | 黄区拉取的完整 URL | pull |
| `NPU_SYNC_LOCAL_BRANCH` | `npu-test` | 黄区落地的本地分支名 | pull |
| `NPU_SYNC_LOG` | `/tmp/npu-run.log` | 日志路径（摘要路径由它派生） | collect |
| `NPU_SYNC_TAIL` | `80` | 摘要里保留的末尾行数 | collect |

### 常见用法

**同时测两套改动**，互不干扰：

```bash
# 蓝区
NPU_SYNC_BRANCH=wip/npu-exp2 ./.agent/sync/push-to-npu.sh
# 黄区
NPU_SYNC_BRANCH=wip/npu-exp2 NPU_SYNC_LOCAL_BRANCH=npu-exp2 ./.agent/sync/pull-on-npu.sh
```

**日志放到有空间的盘**（NPU 机器 `/tmp` 常常很小）：

```bash
NPU_SYNC_LOG=/data/logs/npu-run.log ./.agent/sync/collect.sh pytest ...
```

**摘要留更多上下文**：

```bash
NPU_SYNC_TAIL=300 ./.agent/sync/collect.sh pytest ...
```

---

## 四、故障排查

### 蓝区 push 报 `Authentication failed` / `403`

`origin` 里的 GitHub token 失效或已轮换。重设：

```bash
git remote set-url origin https://github.com/psv666/vllm-omni.git
gh auth login          # 或配 credential helper
```

> 不要再把 token 明文写进 URL —— 它会存进 `.git/config`，任何 `git remote -v` 输出、日志、截图都会泄露。

### 黄区 fetch 卡住 / 超时

黄区到 github.com 的通道断了。先单独确认：

```bash
git ls-remote https://github.com/psv666/vllm-omni.git wip/npu-test
```

能返回一行 sha 就说明通道正常。返回不了就是网络/代理问题，和脚本无关。

### 黄区提示 `error: pathspec 'FETCH_HEAD' did not match`

fetch 那一步没成功（往往被上面的网络问题吞掉了）。单独跑 fetch 看报错。

### 黄区同步完，代码还是旧的

确认蓝区真的推了新的：

```bash
# 蓝区
git ls-remote origin wip/npu-test
# 黄区，两边 sha 应一致
git rev-parse HEAD
```

digest 里的 `commit` 字段就是给这件事用的 —— 每次看一眼，能立刻发现"测的还是上一版"。

### 蓝区推送后发现少了文件

`push-to-npu.sh` 用 `git add -A`，遵守 `.gitignore` 和 `.git/info/exclude`。如果某个新文件没过去，检查是不是被忽略了：

```bash
git check-ignore -v path/to/file
```

---

## 五、安全注意

**`psv666/vllm-omni` 这个 fork 是 public 仓库**，推上去的内容全世界可见。

已经在 `.git/info/exclude` 里屏蔽了这些不该外泄的文件：

```
/CONVERSATION_HISTORY.md     # 含 mentor 相关对话标题、内网主机名、内网路径
/.agent/*                    # 个人笔记
!/.agent/sync/               # 先放行目录这一层，git 才会递归进去
!/.agent/sync/**             # 再放行里面的文件
```

> ⚠️ 排除目录的取消忽略必须写成上面这两行。只写 `!/.agent/sync/**` 不生效 —— git 不会递归进一个被排除的目录，里面的文件根本没机会匹配到取消规则。另外**不要**再往下加一条裸的 `.agent/`：它不带前导 `/`，会匹配任意层级的 `.agent` 目录，把上面的放行整个盖掉（这正是之前 sync 脚本没被推过去的原因）。

**新增含内网信息的文件时，记得同样加一条**，否则会被 `git add -A` 一起推到公开仓库。加完验证：

```bash
git check-ignore -v 新文件路径      # 有输出 = 已屏蔽
```

反过来，**改完忽略规则后要验证该走的确实走了**：

```bash
git check-ignore -v .agent/sync/pull-on-npu.sh   # 无输出 = 会被推过去，这才对
```

推之前想确认这次到底会推什么，看 `push-to-npu.sh` 输出的 diffstat 就行 —— 它列的就是全部内容。

---

## 六、设计说明

几个不那么显然的选择，改脚本前先看一眼。

**为什么用临时 index 而不是 `git stash` 或临时提交？**
`push-to-npu.sh` 把 `GIT_INDEX_FILE` 指到一个临时文件，在那里 `read-tree HEAD` + `add -A` + `write-tree`，再用 `commit-tree` 造一个游离 commit 直接推。全程不碰真实 index 和 working tree，也不在功能分支上留 WIP 提交污染 PR。`git stash create` 做不到 —— 它不包含未跟踪文件。

**为什么黄区用 `checkout -f -B` 而不是 `reset --hard`？**
`reset --hard` 改写的是"当前所在的分支"。黄区常停在 `main`（跟踪 upstream），把 `main` 改写成 scratch commit 之后 log 和 pull 都会乱。`checkout -f -B npu-test` 固定落在专用分支上，其他分支永远不受影响。

**为什么 pull 脚本写死 URL 而不用 `origin`？**
黄区的 `origin` 指向官方仓库 vllm-project/vllm-omni，不是蓝区的 fork。写死完整 URL 后，两边 remote 怎么配都无所谓。fork 是 public，黄区匿名拉取即可，不需要在隔离机器上放 token。

**为什么是 force-push 到固定分支，而不是每次新分支或走 patch？**
scratch 分支的历史没有价值，只要"最新一版"。固定分支 + force-push 让黄区的命令永远是同一条，且 `checkout -f -B` 保证**永远不会有冲突**。走 patch 文件则需要手工搬运、容易 apply 失败，正是这套方案要消灭的麻烦。

**代价**：`wip/npu-test` 上的历史随时会被覆盖，不要在这个分支上做任何有价值的工作。它是一次性的传输载体。用完可以删：

```bash
git push origin --delete wip/npu-test
```
