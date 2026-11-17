#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Called by repro_9258.sh inside the matching, frozen CI image.
set -Eeuo pipefail
cd /opt/vllm-omni
mkdir -p /repro/env /repro/results /repro/server-stderr
export BENCHMARK_DIR=/repro/results
export REPRO_STDERR_DIR=/repro/server-stderr
config_path=tests/dfx/perf/tests/test_qwen3_omni_duplex_omni_duplex_eval.json
export REPRO_CONFIG_PATH="$config_path"

collect_outputs() {
  local status=$?
  trap - EXIT
  set +e
  python - <<'PY'
import json
import os
import shutil
from pathlib import Path
from xml.etree import ElementTree

out = Path('/repro')
config = json.loads(Path(os.environ['REPRO_CONFIG_PATH']).read_text())
for case in config:
    for params in case['benchmark_params']:
        response_root = params.get('duplex_eval_response_root')
        if not response_root:
            continue
        source = Path(response_root)
        if not source.is_dir():
            continue
        for path in source.rglob('*.json'):
            dest = out / 'response-json' / source.name / path.relative_to(source)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
rows = []
for path in sorted((out / 'results').glob('result_*.json')):
    result = json.loads(path.read_text())
    duplex = result.get('omni_duplex_eval') or {}
    accuracy = duplex.get('accuracy') or {}
    rows.append({
        'file': path.name,
        **{k: result.get(k) for k in ('Hardware', 'name', 'num_prompts', 'max_concurrency',
                                    'completed', 'failed', 'timed_out_requests')},
        'published': duplex.get('published'),
        'duplex_retry_count': duplex.get('duplex_retry_count'),
        'accuracy': {k: accuracy.get(k) for k in ('status', 'total', 'evaluated', 'skipped', 'failed')},
    })
summary = {'results': rows}
junit = out / 'pytest.xml'
if junit.exists():
    root = ElementTree.parse(junit).getroot()
    suites = [root] if root.tag == 'testsuite' else root.findall('testsuite')
    summary['pytest'] = {k: sum(int(s.get(k, 0)) for s in suites)
                         for k in ('tests', 'failures', 'errors', 'skipped')}
(out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print('\nResult summary:\n' + json.dumps(summary, indent=2))
PY
  nvidia-smi > /repro/nvidia-smi-after.txt
  printf '%s\n' "$status" > /repro/inside-exit-code.txt
  exit "$status"
}
trap collect_outputs EXIT

[[ "$(git rev-parse HEAD)" == "$EXPECTED_OMNI_SHA" ]] || { echo 'Unexpected mounted checkout.' >&2; exit 2; }
cp "$config_path" /repro/env/original-test-config.json
# Produce a per-run configuration so the pinned source checkout stays pristine.
python - "$config_path" /repro/env/test-config.json <<'PY'
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

source, destination = map(Path, sys.argv[1:])
config = json.loads(source.read_text())
local_models = {}
for variable, role in [('QWEN3_OMNI_MODEL_DIR', 'server_params'), ('JUDGE_MODEL_DIR', 'judge_server_params')]:
    value = os.environ.get(variable)
    if not value:
        continue
    directory = Path(value).resolve(strict=True)
    model_config = directory / 'config.json'
    assert model_config.is_file(), f'{variable}: missing {model_config}'
    weights = list(directory.glob('*.safetensors')) + list(directory.glob('pytorch_model*.bin'))
    assert weights, f'{variable}: no model weights found in {directory}'
    assert all(path.is_file() for path in weights), (
        f'{variable}: unreadable weight symlink in {directory}; '
        'for a Hugging Face cache snapshot, use HF_CACHE_DIR with its complete hub directory'
    )
    for case in config:
        case[role]['model'] = str(directory)
    local_models[variable] = {'container_path': str(directory),
                              'config_sha256': hashlib.sha256(model_config.read_bytes()).hexdigest()}
local_dataset = os.environ.get('DUPLEX_EVAL_DATASET_DIR')
dataset_metadata = {}
if local_dataset:
    directory = Path(local_dataset).resolve(strict=True)
    assert directory.is_dir(), f'Local dataset directory is unavailable: {directory}'
    files = sorted(directory.rglob('*.parquet'))
    assert files and all(path.is_file() for path in files), (
        f'Local dataset needs readable Parquet files: {directory}'
    )
    for case in config:
        for params in case['benchmark_params']:
            params['dataset_path'] = str(directory)
    dataset_metadata = {'container_path': str(directory),
                        'parquet_files': [{'path': str(path.relative_to(directory)), 'bytes': path.stat().st_size}
                                          for path in files]}
if local_models or local_dataset:
    destination.write_text(json.dumps(config, indent=2) + '\n')
else:
    shutil.copyfile(source, destination)
(destination.parent / 'local-models.json').write_text(json.dumps(local_models, indent=2) + '\n')
(destination.parent / 'local-dataset.json').write_text(json.dumps(dataset_metadata, indent=2) + '\n')
print('Local model overrides: ' + json.dumps(local_models))
print('Local dataset override: ' + json.dumps(dataset_metadata))
PY
config_path=/repro/env/test-config.json
export REPRO_CONFIG_PATH="$config_path"
cp vllm_omni/deploy/qwen3_omni*.yaml /repro/env/
cp docker/Dockerfile.ci /repro/env/Dockerfile.ci
python -m pip freeze > /repro/env/pip-freeze.txt
nvidia-smi > /repro/env/nvidia-smi.txt
python - <<'PY' | tee /repro/env/python-runtime.json
import json
import os
import platform
import sys
from urllib.parse import urlsplit
import torch
import vllm
import vllm_omni
from vllm._version import __commit_id__

assert torch.cuda.is_available(), 'CUDA unavailable: check driver and NVIDIA Container Toolkit.'
assert torch.cuda.device_count() == 3, f'Expected 3 visible GPUs, got {torch.cuda.device_count()}'
devices = [{'ordinal': i, 'name': torch.cuda.get_device_name(i),
            'capability': torch.cuda.get_device_capability(i),
            'memory_GiB': round(torch.cuda.get_device_properties(i).total_memory / 2**30, 2)}
           for i in range(3)]
assert all('A100' in d['name'] for d in devices), devices
assert '/opt/vllm-omni/' in vllm_omni.__file__, vllm_omni.__file__
endpoint = urlsplit(os.environ.get('HF_ENDPOINT') or 'https://huggingface.co')
endpoint = endpoint._replace(netloc=endpoint.netloc.rsplit('@', 1)[-1], query='', fragment='').geturl()
print(json.dumps({'python': sys.version, 'platform': platform.platform(),
                  'torch': torch.__version__, 'cuda': torch.version.cuda,
                  'vllm': vllm.__version__, 'vllm_commit': __commit_id__,
                  'omni_file': vllm_omni.__file__, 'devices': devices,
                  'profile': os.environ['REPRO_PROFILE'],
                  'hf_endpoint': endpoint,
                  'hf_hub_offline': os.environ.get('HF_HUB_OFFLINE'),
                  'hf_datasets_offline': os.environ.get('HF_DATASETS_OFFLINE'),
                  'proxy_env_set': [name for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy')
                                    if os.environ.get(name)],
                  'capture_stderr': os.environ.get('CAPTURE_STDERR'),
                  'prefetch': os.environ.get('PREFETCH')}, indent=2))
PY
python .buildkite/cuda/scripts/check_vllm_runtime.py 2>&1 | tee /repro/env/runtime-check.log

if [[ "${PREFETCH:-1}" == 1 ]]; then
  # Download files only; this never starts an Omni engine or a duplex session.
  # The tested engine and its first session are created later in a new process.
  python - <<'PY' 2>&1 | tee /repro/prefetch.log
import json
import os
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
from vllm_omni.benchmarks.duplex.omni_duplex_eval_dataset import load_samples

config = json.loads(Path(os.environ['REPRO_CONFIG_PATH']).read_text())
api = HfApi()
assets = {'models': [], 'datasets': []}
model_ids = sorted({p['model'] for case in config
                    for p in (case['server_params'], case['judge_server_params'])})
for model_id in model_ids:
    if Path(model_id).is_dir():
        print(f'Using mounted local model: {model_id}', flush=True)
        assets['models'].append({'local_path': model_id})
        continue
    info = api.model_info(model_id)
    print(f'Downloading {model_id} at {info.sha}', flush=True)
    snapshot_download(repo_id=model_id, revision=info.sha)
    assets['models'].append({'repo': model_id, 'revision': info.sha})
for case in config:
    for params in case['benchmark_params']:
        dataset = params['dataset_path']
        local_dataset = Path(dataset).is_dir()
        if not local_dataset:
            repo, separator, revision = dataset.partition('@')
            info = api.dataset_info(repo, revision=revision if separator else None)
        print(f'Preparing dataset {dataset}, split={params["duplex_eval_split"]}', flush=True)
        samples = load_samples(dataset, split=params['duplex_eval_split'], limit=1)
        assert samples, 'Dataset returned no samples.'
        if local_dataset:
            assets['datasets'].append({'local_path': dataset, 'split': params['duplex_eval_split']})
        else:
            assets['datasets'].append({'repo': repo, 'revision': info.sha,
                                        'split': params['duplex_eval_split']})
Path('/repro/env/assets.json').write_text(json.dumps(assets, indent=2) + '\n')
PY
fi

# H100/B200 are collection labels in this config; actual hardware is recorded
# independently in the result JSON as A100. Do not replace this with -m A100.
selection=(tests/dfx/perf/scripts/run_benchmark.py --test-config-file "$config_path"
  -m 'full_model and H100 and B200 and cuda and omni and cards_3' --run-level full_model)
python -m pytest --collect-only -q "${selection[@]}" 2>&1 | tee /repro/collection.log
plugin_args=()
if [[ "${CAPTURE_STDERR:-1}" == 1 ]]; then
  export PYTHONPATH="/repro-support${PYTHONPATH:+:$PYTHONPATH}"
  plugin_args=(-p repro_stderr)
fi
set +e
python -m pytest -s -vv --tb=long --junitxml=/repro/pytest.xml \
  "${plugin_args[@]}" "${selection[@]}" 2>&1 | tee /repro/pytest.log
pytest_status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$pytest_status" > /repro/pytest-exit-code.txt
if [[ "$pytest_status" -ne 0 ]]; then exit "$pytest_status"; fi

# A skipped run or an all-failed benchmark is not a successful reproduction check.
python - <<'PY'
import json
from pathlib import Path
from xml.etree import ElementTree

root = ElementTree.parse('/repro/pytest.xml').getroot()
cases = root.findall('.//testcase') if root.tag != 'testcase' else [root]
assert cases, 'No test cases executed.'
assert not any(c.find('skipped') is not None for c in cases), 'A test was skipped.'
results = list(Path('/repro/results').glob('result_*.json'))
assert results, 'No result JSON found.'
for path in results:
    result = json.loads(path.read_text())
    assert result.get('completed') == result.get('num_prompts'), path
    assert result.get('failed') == 0, path
    print(path.name, 'completed=', result['completed'], 'failed=', result['failed'])
PY
