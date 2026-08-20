# Qwen3-Omni Nightly Buildkite 冗余案例分析

> 生成时间：2026-08-20 03:47 UTC  
> 基线：`/data/pengsiv/vllm-omni-main` @ `4cd66787` (`main`)  
> 历史证据：已载入 20 次符合条件的 scheduled nightly；来源 anonymous public web；扫描 2 页，另有 1 次 schedule_unresolved

## 结论摘要

分析了 20 次 scheduled nightly；逐案建议统计为 保留 69、证据不足 7。

静态展开得到 **76 个 pytest/perf 场景**、**11 个模型启动组**，并能静态确认至少 **2458 个请求**（accuracy 等动态数据集请求未估算）。

## Nightly 执行图

| Buildkite job | 类型 | H100 | 场景数 | 启动组 | 请求数 | 成功率 | 失败 | 重试 | flake | P50 | P95 | GPU-hours | 重试 GPU-hours |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| :full_moon: Omni · Function Test with H100 | function | 2 | 51 | 5 | 204 | 60.0% | 8 | 9 | 3 | 97.7 min | 110.2 min | 92.76 | 28.55 |
| :full_moon: Omni · Accuracy Test | accuracy | 2 | 2 | 1 | 0 | 55.0% | 9 | 2 | 0 | 46.8 min | 51.2 min | 35.18 | 3.43 |
| :full_moon: Omni · Perf Test · No Async Chunk | performance | 2 | 4 | 1 | 394 | 100.0% | 0 | 0 | 0 | 37.5 min | 41.0 min | 25.15 | 0.00 |
| :full_moon: Omni · Perf Test · Async Chunk | performance | 2 | 8 | 1 | 788 | 100.0% | 0 | 0 | 0 | 47.2 min | 54.4 min | 32.22 | 0.00 |
| :full_moon: Omni · Perf Test · vLLM Text | performance | 2 | 4 | 1 | 394 | 100.0% | 0 | 0 | 0 | 18.3 min | 20.0 min | 12.22 | 0.00 |
| :full_moon: Omni · Perf Test · Multi-Replica | performance | 3 | 4 | 1 | 662 | 100.0% | 0 | 2 | 2 | 42.3 min | 43.6 min | 46.35 | 4.37 |
| :full_moon: Omni · Multi-Replica Startup Test with 4x H100 | multi-replica-function | 4 | 3 | 1 | 16 | 100.0% | 0 | 0 | 0 | 8.8 min | 10.1 min | 11.98 | 0.00 |

耗时和 GPU-hours 均保持 **job 粒度**；未从日志获得 case 时间戳时不会平均摊派。

## 冗余与相邻覆盖

没有发现可仅凭静态证据认定的完全重复场景。

需要特别避免的误判：

- sync 与 async-chunk 请求相似，但覆盖不同调度/流式数据路径。
- vLLM text perf 与 Omni text-output perf 的 backend、endpoint 和启动方式不同。
- multi-replica 功能测试验证路由与输出；multi-replica perf 验证扩展性能，不能互相替代。
- one-word pronunciation 是随机性烟测；Seed-TTS WER 是数据集准确率门禁，两者只存在相邻覆盖。
- `batch_token_64` 参数 ID 实际配置为 `max_num_batched_tokens=2048`，应单独修正命名，但不构成冗余。

## 逐案建议

完整字段见同目录 CSV。下表只展示需要关注的案例；`证据不足` 不等于建议删除。

| Case | Job | 执行 | 独立故障 | 建议 | 风险/理由 |
|---|---|---:|---:|---|---|
| `tests/e2e/accuracy/qwen3_omni/test_qwen3_omni.py::test_qwen3_omni_seed_tts_wer_bench[omni_server0]` | :full_moon: Omni · Accuracy Test | 20 | 5 | 保留 | 低：历史上发现过独立故障 |
| `tests/e2e/offline_inference/test_qwen3_omni_autoround_w4a16_expansion.py::test_audio_to_text[omni_runner0]` | :full_moon: Omni · Function Test with H100 | 20 | 0 | 证据不足 | 中：无独立故障，但缺少可证明的替代覆盖 |
| `tests/e2e/offline_inference/test_qwen3_omni_autoround_w4a16_expansion.py::test_image_to_text[omni_runner0]` | :full_moon: Omni · Function Test with H100 | 20 | 0 | 证据不足 | 中：无独立故障，但缺少可证明的替代覆盖 |
| `tests/e2e/offline_inference/test_qwen3_omni_autoround_w4a16_expansion.py::test_mix_to_audio[omni_runner0]` | :full_moon: Omni · Function Test with H100 | 20 | 0 | 证据不足 | 中：无独立故障，但缺少可证明的替代覆盖 |
| `tests/e2e/offline_inference/test_qwen3_omni_autoround_w4a16_expansion.py::test_text_to_text[omni_runner0]` | :full_moon: Omni · Function Test with H100 | 20 | 0 | 证据不足 | 中：无独立故障，但缺少可证明的替代覆盖 |
| `tests/e2e/offline_inference/test_qwen3_omni_autoround_w4a16_expansion.py::test_video_to_audio[omni_runner0]` | :full_moon: Omni · Function Test with H100 | 20 | 0 | 证据不足 | 中：无独立故障，但缺少可证明的替代覆盖 |
| `tests/e2e/offline_inference/test_qwen3_omni_autoround_w4a16_expansion.py::test_video_to_text[omni_runner0]` | :full_moon: Omni · Function Test with H100 | 20 | 0 | 证据不足 | 中：无独立故障，但缺少可证明的替代覆盖 |
| `tests/e2e/offline_inference/test_qwen3_omni_modelopt_nvfp4_w4a4.py::test_marlin_fallback_no_nan_collapse[omni_runner0]` | :full_moon: Omni · Function Test with H100 | 12 | 0 | 证据不足 | 未知：少于 20 次有效历史执行 |
| `tests/e2e/online_serving/test_qwen3_omni_expansion.py::test_audio_in_video_001[async_chunk]` | :full_moon: Omni · Function Test with H100 | 12 | 0 | 保留 | 中：历史样本不足，但覆盖独立风险路径 |
| `tests/e2e/online_serving/test_qwen3_omni_expansion.py::test_audio_in_video_002[async_chunk]` | :full_moon: Omni · Function Test with H100 | 13 | 0 | 保留 | 中：历史样本不足，但覆盖独立风险路径 |
| `tests/e2e/online_serving/test_qwen3_omni_expansion.py::test_audio_in_video_002[default]` | :full_moon: Omni · Function Test with H100 | 12 | 0 | 保留 | 中：历史样本不足，但覆盖独立风险路径 |
| `tests/e2e/online_serving/test_qwen3_omni_expansion.py::test_audio_in_video_default_loader_sampling_regression[async_chunk]` | :full_moon: Omni · Function Test with H100 | 13 | 0 | 保留 | 中：历史样本不足，但覆盖独立风险路径 |
| `tests/e2e/online_serving/test_qwen3_omni_expansion.py::test_audio_in_video_default_loader_sampling_regression[default]` | :full_moon: Omni · Function Test with H100 | 12 | 0 | 保留 | 中：历史样本不足，但覆盖独立风险路径 |
| `tests/e2e/online_serving/test_qwen3_omni_expansion.py::test_invalid_audio_format_rejected[default]` | :full_moon: Omni · Function Test with H100 | 13 | 0 | 保留 | 中：历史样本不足，但覆盖独立风险路径 |
| `tests/e2e/online_serving/test_qwen3_omni_expansion.py::test_language_001[async_chunk]` | :full_moon: Omni · Function Test with H100 | 12 | 0 | 保留 | 中：历史样本不足，但覆盖独立风险路径 |
| `tests/e2e/online_serving/test_qwen3_omni_expansion.py::test_large_image_to_text_audio_001[batch_token_64]` | :full_moon: Omni · Function Test with H100 | 15 | 0 | 保留 | 中：历史样本不足，但覆盖独立风险路径 |
| `tests/e2e/online_serving/test_qwen3_omni_expansion.py::test_large_image_to_text_audio_001[default]` | :full_moon: Omni · Function Test with H100 | 15 | 0 | 保留 | 中：历史样本不足，但覆盖独立风险路径 |
| `tests/e2e/online_serving/test_qwen3_omni_expansion.py::test_one_word_prompt_001[async_chunk]` | :full_moon: Omni · Function Test with H100 | 12 | 0 | 保留 | 中：历史样本不足，但覆盖独立风险路径 |
| `tests/e2e/online_serving/test_qwen3_omni_expansion.py::test_one_word_prompt_001[default]` | :full_moon: Omni · Function Test with H100 | 13 | 0 | 保留 | 中：历史样本不足，但覆盖独立风险路径 |
| `tests/e2e/online_serving/test_qwen3_omni_expansion.py::test_speaker_001[async_chunk]` | :full_moon: Omni · Function Test with H100 | 12 | 0 | 保留 | 中：历史样本不足，但覆盖独立风险路径 |
| `tests/e2e/online_serving/test_qwen3_omni_expansion.py::test_speaker_001[default]` | :full_moon: Omni · Function Test with H100 | 13 | 0 | 保留 | 中：历史样本不足，但覆盖独立风险路径 |
| `tests/e2e/online_serving/test_qwen3_omni_expansion.py::test_speaker_002[async_chunk]` | :full_moon: Omni · Function Test with H100 | 13 | 0 | 保留 | 中：历史样本不足，但覆盖独立风险路径 |
| `tests/e2e/online_serving/test_qwen3_omni_expansion.py::test_speaker_002[default]` | :full_moon: Omni · Function Test with H100 | 14 | 0 | 保留 | 中：历史样本不足，但覆盖独立风险路径 |
| `tests/e2e/online_serving/test_qwen3_omni_expansion.py::test_speaker_003[async_chunk]` | :full_moon: Omni · Function Test with H100 | 13 | 0 | 保留 | 中：历史样本不足，但覆盖独立风险路径 |
| `tests/e2e/online_serving/test_qwen3_omni_expansion.py::test_speaker_003[default]` | :full_moon: Omni · Function Test with H100 | 14 | 0 | 保留 | 中：历史样本不足，但覆盖独立风险路径 |
| `tests/e2e/online_serving/test_qwen3_omni_expansion.py::test_text_audio_to_text_audio_001[async_chunk]` | :full_moon: Omni · Function Test with H100 | 12 | 0 | 保留 | 中：历史样本不足，但覆盖独立风险路径 |
| `tests/e2e/online_serving/test_qwen3_omni_expansion.py::test_text_audio_to_text_audio_002[async_chunk]` | :full_moon: Omni · Function Test with H100 | 12 | 0 | 保留 | 中：历史样本不足，但覆盖独立风险路径 |
| `tests/e2e/online_serving/test_qwen3_omni_expansion.py::test_text_audio_to_text_audio_002[batch_token_64]` | :full_moon: Omni · Function Test with H100 | 14 | 0 | 保留 | 中：历史样本不足，但覆盖独立风险路径 |
| `tests/e2e/online_serving/test_qwen3_omni_expansion.py::test_text_image_to_text_audio_001[batch_token_64]` | :full_moon: Omni · Function Test with H100 | 14 | 0 | 保留 | 中：历史样本不足，但覆盖独立风险路径 |
| `tests/e2e/online_serving/test_qwen3_omni_expansion.py::test_text_to_audio_long_output_001[async_chunk]` | :full_moon: Omni · Function Test with H100 | 12 | 0 | 保留 | 中：历史样本不足，但覆盖独立风险路径 |
| `tests/e2e/online_serving/test_qwen3_omni_expansion.py::test_text_to_audio_long_output_001[default]` | :full_moon: Omni · Function Test with H100 | 13 | 0 | 保留 | 中：历史样本不足，但覆盖独立风险路径 |
| `tests/e2e/online_serving/test_qwen3_omni_expansion.py::test_text_to_text_audio_001[async_chunk]` | :full_moon: Omni · Function Test with H100 | 12 | 0 | 保留 | 中：历史样本不足，但覆盖独立风险路径 |
| `tests/e2e/online_serving/test_qwen3_omni_expansion.py::test_text_video_to_text_001[async_chunk]` | :full_moon: Omni · Function Test with H100 | 12 | 0 | 保留 | 中：历史样本不足，但覆盖独立风险路径 |
| `tests/e2e/online_serving/test_qwen3_omni_expansion.py::test_text_video_to_text_audio_001[batch_token_64]` | :full_moon: Omni · Function Test with H100 | 15 | 0 | 保留 | 中：历史样本不足，但覆盖独立风险路径 |
| `tests/e2e/online_serving/test_qwen3_omni_expansion.py::test_text_video_to_text_audio_001[default]` | :full_moon: Omni · Function Test with H100 | 15 | 0 | 保留 | 中：历史样本不足，但覆盖独立风险路径 |
| `tests/e2e/online_serving/test_qwen3_omni_multi_replicas.py::test_mixed_modal_stream_batch_generates_text_and_audio[omni_server0]` | :full_moon: Omni · Multi-Replica Startup Test with 4x H100 | 20 | 0 | 保留 | 低：覆盖独立配置或质量维度 |
| `tests/e2e/online_serving/test_qwen3_omni_multi_replicas.py::test_text_only_batch_uses_multi_replica_talker[omni_server0]` | :full_moon: Omni · Multi-Replica Startup Test with 4x H100 | 20 | 0 | 保留 | 低：覆盖独立配置或质量维度 |
| `tests/e2e/online_serving/test_qwen3_omni_multi_replicas.py::test_text_to_audio_stream_batch_uses_multi_replica_vocoder[omni_server0]` | :full_moon: Omni · Multi-Replica Startup Test with 4x H100 | 20 | 0 | 保留 | 低：覆盖独立配置或质量维度 |
| `tests/dfx/perf/scripts/run_benchmark.py::test_performance_benchmark[test_qwen3_omni_3gpu_replica2-random-mm]` | :full_moon: Omni · Perf Test · Multi-Replica | 20 | 0 | 保留 | 低：覆盖独立配置或质量维度 |
| `tests/dfx/perf/scripts/run_benchmark.py::test_performance_benchmark[test_qwen3_omni_3gpu_replica2-random-mm_1]` | :full_moon: Omni · Perf Test · Multi-Replica | 20 | 0 | 保留 | 低：覆盖独立配置或质量维度 |
| `tests/dfx/perf/scripts/run_benchmark.py::test_performance_benchmark[test_qwen3_omni_3gpu_replica2-random-mm_2]` | :full_moon: Omni · Perf Test · Multi-Replica | 20 | 0 | 保留 | 低：覆盖独立配置或质量维度 |
| `tests/dfx/perf/scripts/run_benchmark.py::test_performance_benchmark[test_qwen3_omni_3gpu_replica2-random]` | :full_moon: Omni · Perf Test · Multi-Replica | 20 | 0 | 保留 | 低：覆盖独立配置或质量维度 |
| `tests/dfx/perf/scripts/run_benchmark.py::test_performance_benchmark[test_qwen3_omni_vllm_text-random-mm]` | :full_moon: Omni · Perf Test · vLLM Text | 20 | 0 | 保留 | 低：覆盖独立配置或质量维度 |
| `tests/dfx/perf/scripts/run_benchmark.py::test_performance_benchmark[test_qwen3_omni_vllm_text-random-mm_1]` | :full_moon: Omni · Perf Test · vLLM Text | 20 | 0 | 保留 | 低：覆盖独立配置或质量维度 |
| `tests/dfx/perf/scripts/run_benchmark.py::test_performance_benchmark[test_qwen3_omni_vllm_text-random-mm_2]` | :full_moon: Omni · Perf Test · vLLM Text | 20 | 0 | 保留 | 低：覆盖独立配置或质量维度 |
| `tests/dfx/perf/scripts/run_benchmark.py::test_performance_benchmark[test_qwen3_omni_vllm_text-random]` | :full_moon: Omni · Perf Test · vLLM Text | 20 | 0 | 保留 | 低：覆盖独立配置或质量维度 |

## 优化基线与灰度验收

当前没有案例通过自动删除门槛，优化后的安全基线与现状相同，预计节省为 0 GPU-hours。

| 指标 | 当前 20 次基线 | 安全优化后 |
|---|---:|---:|
| nightly job 数 | 7 | 7 |
| 模型启动组 | 11 | 11 |
| 累计 GPU-hours | 255.86 | 255.86 |
| 平均 GPU-hours/nightly | 12.79 | 12.79 |
| P95 关键路径 | 110.2 min | 110.2 min |

灰度时保留原集合为对照，连续运行 7 次精简版 nightly。只有同时满足以下条件才落地：

1. 历史回放仍能捕获全部已知独立故障。
2. 关键覆盖维度至少保留一个 nightly 场景。
3. 没有新增连续 flake。
4. GPU-hours 或关键路径下降至少 15%。

回滚方式是恢复候选案例原有 pytest target/参数化；本分析阶段不修改 nightly YAML。

## 方法、限制与来源

- 仅分析 `.buildkite/cuda/test-nightly.yml` 中指定的 Qwen3-Omni H100 功能、accuracy、perf 和 multi-replica job。
- 匿名公开样本必须同时满足 `branch=main`、`source=schedule`、精确消息 `Scheduled nightly build` 且不是 PR；公开 build JSON 不暴露 env 或 schedule 定义。
- job、重试、日志与 artifact 元数据来自 Buildkite 页面使用的匿名 web data 端点；不使用 token，端点属于前端实现，未来可能变化。
- 每次 attempt 的 GPU 数从该 build commit 的 nightly YAML 解析；历史提交不可用时 GPU-hours 留空。
- 静态来源：[nightly YAML](../../../.buildkite/cuda/test-nightly.yml)、[Qwen3-Omni expansion tests](../../../tests/e2e/online_serving/test_qwen3_omni_expansion.py)、[perf configs](../../../tests/dfx/perf/tests/) 和 [分析工具](../../../tools/nightly/analyze_qwen3_omni_nightly.py)。
- Buildkite API 的端点与字段依据见 [API 调研记录](qwen3_omni_nightly_buildkite_api_research.md)。

### 收集告警

- Broad tests/e2e collection failed (ERROR tests/e2e/features/rlhf_test/test_verl_omni_e2e.py); used model-reference fallback with 5 files
