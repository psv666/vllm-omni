# Issue #9258: A100 reproduction

这组脚本用于在 A100 上复测 [#9258](https://github.com/vllm-project/vllm-omni/issues/9258)
报告的首次 duplex session 打开期间 WebSocket 关闭问题。运行所需设置集中在
[config.yaml](config.yaml)，修改本机路径后直接运行，无需逐项 export。

| 模式 | vLLM-Omni 提交 | vLLM | 工作负载 |
| --- | --- | --- | --- |
| `ci` | `22721f0acdfe129ba2a8937057dd321efecf8aa1` | 0.33.0 | PR_correction，120 条，并发 8 |
| `original` | `aa3fd1d9b7f5c7191dec11975ac0aa7c4c4a18c5` | 0.32.0 | PR_correction，3 条，并发 1 |

两种模式都只运行 `test_qwen3_omni_duplex_omni_duplex_eval.json` 中的 `PR_correction`
参数子用例，与 issue 中失败的请求路径对应。`original` 保留 1 条 warmup；`ci` 保留 8 条
warmup，并启动独立的评测模型。脚本生成本轮配置时筛选参数，不修改固定版本的源码配置。

`ci` 的固定版本在 H100 [nightly #19773](https://buildkite.com/vllm/vllm-omni/builds/19773#01a1deaa-a6be-4a2d-b3f5-b3889d75903b)
中获得 `1 passed`、120 个请求成功、0 个请求失败。A100 模型测试结果仍待收集。

## 获取和配置

在 A100 服务器上执行：

```bash
git clone --single-branch --branch codex/repro-9258-a100 \
  https://github.com/psv666/vllm-omni.git vllm-omni-9258-repro
cd vllm-omni-9258-repro
nano tools/repro_9258/config.yaml
```

已有仓库时先 `git pull --ff-only`。配置文件中需要填写的主要内容是：

```yaml
paths:
  repro_root: /home/pengsiv/vllm-omni-9258-repro/issues/issue9258-a100
  hf_cache_dir: ""
  qwen3_omni_model_dir: /home/pengsiv/models/Qwen3-Omni-30B-A3B-Instruct
  judge_model_dir: /home/pengsiv/models/Qwen2.5-7B-Instruct
  duplex_eval_dataset_dir: /home/pengsiv/datasets/Omni-DuplexEval

profiles:
  ci:
    gpus: [0, 1, 2]
  original:
    gpus: [3, 4, 5]
    repro_root: /home/pengsiv/vllm-omni-9258-repro/issues/issue9258-a100-original
```

示例模型和数据路径需要替换成实际目录。三个输入目录在离线模式下都必须填写。
路径可以是绝对路径或相对于 YAML 文件所在目录的路径；`~/` 使用宿主用户的家目录。
`hf_cache_dir` 留空时，两种模式共用 `paths.repro_root/hf-cache`。
各模式可设置独立的 `repro_root`，以分开源码快照和实验日志。

其余设置也在同一 YAML 中：

- `profile`：不传命令行模式时运行的默认模式。
- `offline`：默认 `true`，禁止脚本自动获取缺失源码和镜像，并关闭 Hugging Face 在线访问。
- `prefetch`：默认 `true`，启动模型服务前先检查本地输入是否可读。
- `capture_stderr`：默认 `true`，保存测试 fixture 内部消费的服务端 stderr。
- `log_level`：默认 `INFO`；需要更详细的服务端输出时可设为 `DEBUG`。
- `source_repo`、`hf_endpoint`：在线模式使用的 Git 仓库和 Hugging Face 地址。

YAML 中的设置优先于终端中先前 export 的同名变量。
离线模式遇到未填写的输入、缺少源码或镜像时会报错，不会自动联网补齐。
已有镜像始终直接复用。只有显式设为 `offline: false`，才允许获取缺失的资源。

## 运行

```bash
# 只检查并打印所选模式的配置，不运行 GPU 测试。
bash tools/repro_9258/repro_9258.sh --check-config

# 使用 YAML 中的默认 profile。
bash tools/repro_9258/repro_9258.sh

# 显式选择模式；GPU 和目录仍从同一个 YAML 读取。
bash tools/repro_9258/repro_9258.sh ci
bash tools/repro_9258/repro_9258.sh original
```

另存配置文件时，可以使用 `--config /path/to/another.yaml`。
运行需要 Linux x86_64、Bash、Git、Docker、NVIDIA Container Toolkit、`nvidia-smi` 和 `tar`。
每组使用三张 A100 80GB：容器逻辑 GPU 0 跑 Thinker，GPU 1 跑 Talker/Code2Wav，GPU 2 跑评测模型。

配置解析优先使用宿主已有的 Python 3.8+ 和 PyYAML；未安装 PyYAML 时，自动使用已经缓存的
CI 镜像解析，解析容器的网络关闭，不要求额外安装宿主依赖。
至少需要这两种解析环境中的一种。CI 镜像和源码提交固定在脚本中。

两个模式可以在不同终端运行，分别使用 YAML 中配置的 GPU 和目录。
并行运行仍会共享 CPU、内存和磁盘，冷启动超时需要再单独运行确认。

## 本地模型和数据

模型目录应包含 `config.json`、完整权重及 tokenizer/processor 配置。
脚本将模型目录只读挂载进容器，并生成本次运行的测试配置。
已有 Hugging Face 缓存时，`hf_cache_dir` 应指向包含 `hub/` 的缓存根目录。

数据集若来自 Hugging Face 缓存，`duplex_eval_dataset_dir` 填包含 `blobs/`、
`refs/`、`snapshots/` 等目录的仓库缓存根目录，例如：

```yaml
paths:
  duplex_eval_dataset_dir: /data/models/hub/datasets--Hothan--Omni-DuplexEval
```

脚本只读挂载整个缓存，保留快照到 `blobs/`（或 `trees/`）的相对链接。
不要只挂载含有目录外链接的 `snapshots/main/`。独立的本地目录也可使用，前提是
所需 `data/<split>-00000-of-000NN.parquet` 分片完整，链接目标都在挂载目录内。
已转换的 datasets Arrow 缓存目录不能作为这个设置的输入。

`ci` 只选择固定 revision `de63329f76daa5e0c604363ef6a2c383dfba45f6` 的快照，
缺失时报错，不下载，也不混入其他快照。`original` 原始配置未固定数据集 revision，
按 `refs/main`、`snapshots/main/`、唯一快照的顺序选择；存在多个快照且无法确定时会报错。
两组实验应使用同一份模型和数据；可在每次运行的 `env/local-dataset.json` 核对所选快照。

脚本在 `env/local-dataset-view/README.md` 生成分片映射，直接引用选中快照已有的文件，
不复制或改写数据文件。映射保留 `PR_correction` 等分片身份，避免缓存根目录被合并为
通用 `train`。两种模式都只需要 `PR_correction` 分片。
启动模型前检查所需分片数量、链接和 Parquet 文件首尾。
默认 `prefetch: true` 时，还通过原版数据加载器预读。
Hugging Face 可能打印 `Downloading data` 进度条，此处是在准备已经挂载的本地文件。

源码存放在所选 `repro_root/snapshots/<提交>/`。脚本优先从当前完整 Git 仓库准备独立快照；
只有在线模式才会在本地缺少提交时浅拉取固定源码。完成检出后才会发布快照目录，以支持中断重跑。
旧脚本的 `repos/` 不参与新版源码准备，`hf-cache/` 仍可继续复用。

## 日志和诊断包

每组输出位于所选 `repro_root/runs/<模式>-<UTC时间>-<随机后缀>/`。
脚本会打印 `Evidence:` 绝对路径；容器结束后会打印 `Evidence archive:` 路径。

- `evidence.tar.gz`：本次诊断包，包含以下日志和配置。
- `diagnostics.txt`：可直接传出的纯文本诊断资料，包含完整容器日志、服务端 stderr、
  启动命令、退出码、配置、环境和结构化结果。不会只截取错误行或日志末尾。
- `run-config.json`：本次模式实际生效的 YAML 设置，HTTP 地址中的认证和查询参数已去除。
- `pytest.log`、`pytest.xml`、`summary.json`：客户端错误、测试与请求结果。
- `server-stderr/`：服务端 stderr 和对应启动命令。
- `container.log`：容器初始化及测试输出。
- `env/`、GPU 相关文本文件：环境、依赖、原始与实际测试配置、本地输入信息。
- `results/`、`response-json/`：benchmark 结果及响应诊断 JSON。

模型权重和输入数据不会打包。完整响应媒体保留在源码快照的 `tests/dfx/perf/results/`。
归档先在运行目录外生成，再移入本次目录。容器失败后正常退出也会打包，并保留实验退出码。
只能传文本时，提供 `diagnostics.txt` 即可。它不包含模型、输入数据或响应媒体。
已有实验完成后，可从原有日志生成这份文本，无需重跑测试：

```bash
bash tools/repro_9258/collect_text.sh /absolute/path/to/finished/run
```

将参数替换成该轮控制台 `Evidence:` 打印的目录。导出命令会打印 `Text diagnostics:` 路径；
新启动的实验结束后会自动导出。`container.log` 已含 pytest 的控制台输出，文本中不再重复
拼接 `pytest.log`。单独保存的日志文件仍在原位置。

即便 pytest 通过，也应查看 warmup 中的 `Realtime WebSocket closed before the requested event arrived`，
以及服务端对应的 `duplex open timed out` 和异常栈。

## 验证范围

脚本由 OpenAI Codex 辅助编写。本地验证覆盖配置解析、输入路径替换、离线准备分支、
源码快照获取与中断恢复、stderr 采集、诊断包完整性及退出码。数据集准备还使用
`datasets==3.6.0` 和两种模式固定版本的数据加载器验证了真实 Parquet 的分片身份。
完整 A100 模型测试仍待收集。

旧 issue 的代码版本字段填写 `aa3fd1d9`，`collect_env` 则显示 `706f1dd12`；`original` 按前者复测。
旧配置的数据集未固定 revision，因此本次运行不能称为对当时全部环境字节的精确复刻。
