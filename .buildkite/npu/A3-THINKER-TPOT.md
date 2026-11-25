# Qwen3-Omni A3 Thinker TPOT 诊断

在 A3 上执行这个 recipe，判断 no-chunk 长文本负载的并发回归能否复现。复现后，在同一个容器和同一组 NPU 上用第二轮计时缩小范围。每轮只回传脚本的摘要，完整日志和 trace 留在 A3。

## 1. 准备容器和代码

优先使用 `quay.io/atlas-ci/vllm-ascend:v0.34.0-a3`。`docker images` 显示的是镜像。用以下命令找已有的容器并查看本地镜像记录的 digest。

```bash
image=quay.io/atlas-ci/vllm-ascend:v0.34.0-a3
docker ps -a --filter "ancestor=$image" --format '{{.Names}}	{{.Status}}'
docker image inspect "$image" --format '{{json .RepoDigests}}'
```

- 按 A3 本机的资源分配规则取得设备。使用具备诊断仓库、模型缓存和至少 3 个可见逻辑 NPU 的实验容器。Thinker 用 2 个逻辑设备做 TP，Talker 和 Code2Wav 共用第 3 个。
- 如果没有可用容器，优先查找本机已有的 A3 启动手册或启动脚本，用本地镜像启动独立实验容器。沿用本机的驱动挂载和设备分配方式。若找不到规则，回报所缺的启动入口。整个实验用 v0.34，v0.33 留给后续运行栈对照。
- 基础镜像 tag 只说明本机有候选环境。CI 在基础镜像上还安装了 Omni 及依赖。确认容器中的 Python 能收集 benchmark。如果缺包，优先使用本机已有的可用环境或获准的离线依赖来源，无法补齐时回报缺失的依赖。
- 在现有仓库执行下列命令。选一个独立工作树路径，确保该工作树在实验容器内可读。如果分支已经在容器内检出，直接进入其根目录。

```bash
git fetch https://github.com/psv666/vllm-omni.git codex/a3-thinker-tpot-diagnostics
git worktree add --detach ../omni-a3-diag FETCH_HEAD
cd ../omni-a3-diag
git rev-parse HEAD
```

## 2. 预检和复现

以下命令在实验容器内、诊断工作树根目录执行。保存 `run_dir` 的实际路径，后续命令使用同一个目录。

```bash
run_dir=$(mktemp -d /tmp/qwen3-omni-a3.XXXXXXXX)
recipe=.buildkite/npu/scripts/qwen3_omni_a3_triage.py
runtime_args=()
printf 'Local results directory: %s\n' "$run_dir"
```

- 默认从 Hugging Face 缓存定位 `26291f793822fb6be9555850f06dfe95f2d7e695`。模型单独挂载时，在执行任何一轮前用 `runtime_args+=(--model-dir "$model_dir")` 传入已核对的本机绝对路径。脚本校验索引、权重分片和 revision 记录。
- 有可核实的基础镜像 `RepoDigest` 时，用 `runtime_args+=(--base-image-digest "$base_digest")` 传入其中的 `sha256:…`。镜像 ID 不能用作该参数。Buildkite #12664 的基础镜像 digest 为 `sha256:fdcede880eeba2b93d0dc3fc8532d0cb88dd1c2586f5f61cff7e923b7f55cd69`。来源无法核对时省略该参数，结果按现场运行栈解读。
- 在下列预检成功后收集 pytest 用例。容器身份、设备和运行栈以脚本输出为准。如果预检或收集失败，保留本机日志，回报失败步骤和具体缺项，待环境就绪后用新目录重试。

```bash
python "$recipe" check --output "$run_dir/check" "${runtime_args[@]}"
```

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  python -m pytest --collect-only -q -m npu \
  tests/dfx/perf/scripts/run_benchmark.py \
  --test-config-file tests/dfx/perf/tests/test_qwen3_omni_no_async_chunk.json
```

用 `reproduce` 启动第一轮。它先跑 `c=8, n=32` 和 `c=16, n=64`。如果环境与 CI 基础镜像匹配且结果接近正常，脚本会自动用 CI 的 `c=1 → 4 → 8 → 16` 顺序再测。等待脚本结束并读 `summary.json`，日志可在 `reproduce/initial/run.log` 和可能存在的 `reproduce/ci-order/run.log` 里查看。

```bash
python "$recipe" reproduce --output "$run_dir/reproduce" "${runtime_args[@]}"
```

## 3. 复现后再采样

只有第一轮 `run.verdict` 是 `reproduced` 时才继续。确认第一轮测试服务已经退出。用同一个容器、可见设备和 `runtime_args` 运行第二轮。脚本从首轮摘要恢复实际复现的并发顺序，并核对运行栈。

```bash
python "$recipe" profile \
  --first-pass "$run_dir/reproduce/summary.json" \
  --output "$run_dir/profile" "${runtime_args[@]}"
```

- 首轮为 `near_normal` 或 `inconclusive` 时，回传首轮摘要。若是 `invalid`，先按 `run.errors` 和本地日志检查环境与请求校验。
- 第二轮采样以 `run.verdict` 仍为 `reproduced` 且 `probe_gaps` 为空为完整取证条件。若有缺项，照实回传。主机计时用于区分调度、执行器和 worker 路径，设备端算子与 HCCL 归因需要设备侧证据。

### 从已完成的第二轮恢复摘要

如果第二轮报 EngineCore trace 缺失且摘要没有 worker 计时，拉取诊断分支的更新，在新工作树执行下列只读命令。`run_dir` 仍指向原实验目录。命令不启动服务或使用 NPU，已有的 trace 和摘要文件保留原样。

```bash
python .buildkite/npu/scripts/qwen3_omni_a3_triage.py summarize-trace \
  --trace-dir "$run_dir/profile/instrumented/trace"
```

### 核对 batch 8、9 和 16 的图分派

如果已在同一个 A3 运行栈上复现，并发现实际 decode batch 从 8 到 9 进入慢档，更新诊断分支后可直接复用提交 `2f0207881` 产生的原首轮摘要。用原先的 `runtime_args` 跑一次更新后的 profile，选择新输出目录。

```bash
python "$recipe" profile \
  --first-pass "$run_dir/reproduce/summary.json" \
  --output "$run_dir/profile-graphs" "${runtime_args[@]}"
python "$recipe" summarize-graphs --trace-dir "$run_dir/profile-graphs/instrumented/trace"
```

回传 `summarize-graphs` 的摘要和本轮 profile 的 verdict。图数据来自 runner 已选定的执行描述符，包含实际调度 token 数、模式和图的 token/request 尺寸。输入和 attention 准备、模型调用记录的是主机耗时。`forward_hook` 标出记录的是 graph manager 调用、`_call_model` 或未采到的转发路径。

## 4. 回传

每轮复制一段 `A3_SUMMARY_BEGIN` 到 `A3_SUMMARY_END` 的 stdout 内容，包含两行边界，单段上限 4 KB。预检阻塞时只回传预检摘要和可安全分享的缺项。原始日志、模型数据和 trace 保存在 `run_dir`，通过 A3 本机规则结束本次实验占用。

历史对照是 [正常样本 #11820](https://buildkite.com/vllm/vllm-omni-npu-ci/builds/11820#01a1e73e-233a-4198-873b-482381ec903b) 和 [异常样本 #12664](https://buildkite.com/vllm/vllm-omni-npu-ci/builds/12664#01a20476-6207-4590-97f8-fee4728c5107)。


## 图尺寸 A/B 对照

当协调端派发图尺寸 A/B 时，在原实验容器、同一工作树的最新指定提交、同一组设备上执行这条命令。沿用原先的 `runtime_args`，包括原模型路径和可核对的基础镜像 digest。`graph-size-ab` 自行生成配置、串行运行两组、校验真实图分派并打印摘要。

```bash
python .buildkite/npu/scripts/qwen3_omni_a3_triage.py graph-size-ab \
  --output "$run_dir/graph-size-ab" "${runtime_args[@]}"
```

A 保留 Thinker 的 8-token bucket。B 只从 Thinker 的 `prefill_cudagraph_capture_token_sizes` 移除尺寸 8。两组均为 no-chunk `c=8, n=32`。脚本仅用实测 `batch=8`、8 个调度 token 的 PIECEWISE 步判定，要求 A 选中 8-token 图、B 选中 16-token 图。回写 stdout 的完整摘要。若脚本给出 `invalid`，回写摘要并保留日志，等待本地协调端处理。


## 同一 16-token 图的设备侧对照

在原 A3 容器和同一组 NPU 上用协调端给的提交执行以下命令。脚本依次运行 `c=8,n=8` 和 `c=9,n=9`，保持 Thinker 的 PIECEWISE 图尺寸一致。每组经过两条 warmup 请求后，用原 CI 的输入和输出长度测一批。两个 TP worker 各在连续命中目标形状后采四步 NPU trace。

```bash
python .buildkite/npu/scripts/qwen3_omni_a3_triage.py device-compare \
  --output "$run_dir/device-compare" "${runtime_args[@]}"
```

回写完整 `A3_SUMMARY`。`slow_step_reproduced` 标记缩小后的负载是否再现慢档。摘要包含同一采样窗内的设备活跃时间并集、设备时间跨度、识别到的通信占用和 top 算子。算子累计时间可能重叠，采样耗时用于定位。若采样不足、分派不符或 profiler 导出失败，回写现有摘要和缺项。所有 CSV 和 trace 留在 A3。
