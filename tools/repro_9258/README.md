# Issue #9258: A100 reproduction

这组脚本用于在 A100 上复测 [#9258](https://github.com/vllm-project/vllm-omni/issues/9258)
报告的首次 duplex session 打开期间 WebSocket 关闭问题。脚本从 Git 获取源码，使用与固定提交
对应的 CI 镜像启动全新服务，并保存请求结果和服务端日志。

| 模式 | vLLM-Omni 提交 | vLLM | 工作负载 |
| --- | --- | --- | --- |
| `ci` | `22721f0acdfe129ba2a8937057dd321efecf8aa1` | 0.33.0 | PR_correction，120 条，并发 8 |
| `original` | `aa3fd1d9b7f5c7191dec11975ac0aa7c4c4a18c5` | 0.32.0 | PR_correction 3 条、RTD_counting 2 条，并发 1 |

`ci` 使用的版本在 H100 [nightly #19773](https://buildkite.com/vllm/vllm-omni/builds/19773#01a1deaa-a6be-4a2d-b3f5-b3889d75903b)
中获得 `1 passed`、120 个请求成功、0 个请求失败。这里尚未取得 A100 执行结果。

## 环境要求

- Linux x86_64，Bash、Git、Docker、NVIDIA Container Toolkit、`nvidia-smi`、`tar`。
- 三张空闲的 A100 80GB。容器逻辑 GPU 0 跑 Thinker，GPU 1 跑 Talker/Code2Wav，GPU 2 跑评测模型。
- 能拉取 Git 仓库和 `public.ecr.aws` 镜像，并能访问 Hugging Face 模型和数据集。
- 建议预留约 200–300GB 磁盘，用于两个镜像、共享模型缓存和输出。

镜像中的 CUDA 版本为 13。脚本会在模型测试前检查宿主驱动可用性、GPU 数量、vLLM 版本及编译扩展。
如果预检失败，应先处理环境；这类失败本身不能算复现 #9258。

## 通过 Git 获取并运行

在 A100 服务器上执行：

```bash
git clone --single-branch --branch codex/repro-9258-a100 \
  https://github.com/psv666/vllm-omni.git vllm-omni-9258-repro
cd vllm-omni-9258-repro

# 先确认选中的三张物理卡空闲；根据实际情况修改 GPUS。
nvidia-smi
GPUS=0,1,2 bash tools/repro_9258/repro_9258.sh ci

# 等上一组结束，再运行旧版本。
GPUS=0,1,2 bash tools/repro_9258/repro_9258.sh original
```

脚本默认把源码快照、模型缓存和输出放在 `~/issue9258-a100/`。指定其他磁盘时，两组使用同一个目录：

```bash
REPRO_ROOT=/data/issue9258-a100 \
GPUS=0,1,2 bash tools/repro_9258/repro_9258.sh ci
```

源码默认从 `psv666/vllm-omni` 获取。若服务器使用 SSH Git 地址，可设置：

```bash
REPRO_REPO_URL=git@github.com:psv666/vllm-omni.git \
GPUS=0,1,2 bash tools/repro_9258/repro_9258.sh ci
```

`HF_CACHE_DIR` 可指定已有 Hugging Face 缓存目录。`HF_ENDPOINT`、`HF_TOKEN`、代理设置和
`VLLM_LOGGING_LEVEL` 会在设置时传入容器。脚本不会主动打印 token 值。

## 指定已有模型路径

普通模型目录可以通过下面两个参数分别指定，路径替换成服务器上的实际位置：

```bash
export REPRO_ROOT=/home/pengsiv/issue9258-a100
export QWEN3_OMNI_MODEL_DIR=/home/pengsiv/models/Qwen3-Omni-30B-A3B-Instruct
export JUDGE_MODEL_DIR=/home/pengsiv/models/Qwen2.5-7B-Instruct

GPUS=0,1,2 bash tools/repro_9258/repro_9258.sh ci
# 上一组结束后，使用同一组模型目录运行旧版。
GPUS=0,1,2 bash tools/repro_9258/repro_9258.sh original
```

这两个目录应各自包含 `config.json`、tokenizer/processor 配置及完整权重文件。
脚本会只读挂载到容器，生成本次运行的测试配置，并直接使用本地权重。
两个参数可以独立设置；未指定的模型仍按原 Hugging Face ID 加载。
数据集仍按原测试配置准备。比较两个代码版本时，应使用相同的模型文件。

若已有的是 Hugging Face 缓存，应把 `HF_CACHE_DIR` 设置为包含 `hub/` 的缓存根目录，例如：

```bash
# 该目录中应有 hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/ 等目录。
HF_CACHE_DIR=/home/pengsiv/huggingface \
GPUS=0,1,2 bash tools/repro_9258/repro_9258.sh ci
```

缓存的 `snapshots/<revision>/` 通常通过符号链接引用同级 `blobs/` 中的权重；这种布局需要挂载完整缓存。
本次配置保存在 `env/test-config.json`，原配置保存在 `env/original-test-config.json`。

默认先下载模型与数据集文件；该过程不会启动 Omni 引擎或创建 duplex session。
每次测试都会创建新容器并启动新服务。缓存已准备好时可设置 `PREFETCH=0`。

## 测试和日志

测试执行原 JSON 配置，并保留 `-m "full_model and H100 and B200 and cuda and omni and cards_3"`。
H100/B200 是用例的收集标签；这段实现只检查 CUDA 卡数，不据此排除 A100。
结果 JSON 会独立记录实际设备为 A100。

默认加载 `repro_stderr.py`，将 fixture 原本在内部消费的服务端 stderr 同时保存到文件。
该插件只增加诊断输出。需要关闭这一层采集时，设置 `CAPTURE_STDERR=0`。

每组输出位于 `~/issue9258-a100/runs/<模式>-<时间>-<随机后缀>/`：

- `summary.json`：请求成功、失败、超时、评测和 pytest 执行情况。
- `pytest.log`、`pytest.xml`：测试日志和 JUnit 结果。
- `server-stderr/`：服务端 stderr 和对应启动命令。
- `results/`、`response-json/`：benchmark 结果及响应诊断 JSON。
- `env/`：软件版本、GPU、依赖、测试配置、模型和数据集 revision。
- `evidence.tar.gz`：本组诊断包；完整响应媒体保留在相应源码快照的 `tests/dfx/perf/results/`。

即便 pytest 通过，也应查看 warmup 是否包含
`Realtime WebSocket closed before the requested event arrived`。
还应检查服务端的 `duplex open timed out`、异常栈及对应时间。
脚本保留测试退出码，并额外拒绝将 skipped 或请求未全部成功的运行算作通过。

## 验证范围

这组脚本由 OpenAI Codex 辅助编写。本地已检查 Bash/Python 语法、stderr 采集及退出码保留；
尚未在 A100 上执行。镜像固定到 digest，两个源码提交使用各自对应的运行时。

旧 issue 的版本记录存在差异：代码版本字段填写 `aa3fd1d9`，`collect_env` 则显示 `706f1dd12`。
`original` 按前者复测。旧配置中的数据集也未固定 revision；脚本记录本次实际解析的模型与数据集
revision，因此不能把这组运行称为对当时全部环境字节的精确复刻。
