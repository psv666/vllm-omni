#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Edit config.yaml, then run: bash tools/repro_9258/repro_9258.sh [ci|original]
# Optional: --config FILE, --check-config. Configuration comes from YAML.
set -Eeuo pipefail

support_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
config_file="$support_dir/config.yaml"
profile_override=""
check_config=0
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    ci|original)
      [[ -z "$profile_override" ]] || { echo 'Select one profile.' >&2; exit 2; }
      profile_override=$1; shift ;;
    --config)
      [[ "$#" -ge 2 ]] || { echo '--config requires a YAML path.' >&2; exit 2; }
      config_file=$2; shift 2 ;;
    --check-config) check_config=1; shift ;;
    -h|--help)
      echo "Usage: $0 [ci|original] [--config FILE] [--check-config]"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -f "$config_file" ]] || { echo "Missing YAML configuration: $config_file" >&2; exit 2; }
config_dir="$(cd -- "$(dirname -- "$config_file")" && pwd -P)"
config_file="$config_dir/$(basename -- "$config_file")"

read_yaml() {
  local candidate parser_image
  local parser_args=(--base-dir "$config_dir" --home-dir "$HOME")
  if [[ -n "$profile_override" ]]; then parser_args+=(--profile "$profile_override"); fi
  for candidate in python3 python; do
    if command -v "$candidate" >/dev/null && "$candidate" -c 'import sys, yaml; assert sys.version_info >= (3, 8)' >/dev/null 2>&1; then
      "$candidate" "$support_dir/load_config.py" "$config_file" "${parser_args[@]}"
      return
    fi
  done
  # The cached CI images already contain PyYAML; no host package installation
  # or network access is needed to read the configuration.
  command -v docker >/dev/null || { echo 'YAML needs local Python/PyYAML or a cached CI image.' >&2; return 2; }
  for parser_image in \
    public.ecr.aws/q9t5s3a7/vllm-ci-test-repo@sha256:dc76851796905fa258a99edb0a330a7bbe2f7e74c2639888982b590beb35a830 \
    public.ecr.aws/q9t5s3a7/vllm-ci-test-repo@sha256:34f3bf8015bae3450de321d72e7de72ab4366ef496172ca3d65ce41e95de0632; do
    if docker image inspect "$parser_image" >/dev/null 2>&1; then
      docker run --rm --network=none --entrypoint python \
        --mount "type=bind,src=$support_dir,dst=/repro-support,readonly" \
        --mount "type=bind,src=$config_file,dst=/repro-config.yaml,readonly" \
        "$parser_image" /repro-support/load_config.py /repro-config.yaml "${parser_args[@]}"
      return
    fi
  done
  echo 'Reading YAML requires local Python/PyYAML or one already cached CI image; no download was attempted.' >&2
  return 2
}
config_env="$(mktemp)"
if ! read_yaml > "$config_env"; then rm -- "$config_env"; exit 2; fi
while IFS= read -r -d '' name && IFS= read -r -d '' value; do
  export "$name=$value"
done < "$config_env"
rm -- "$config_env"
if [[ "$check_config" -eq 1 ]]; then printf '%s\n' "$REPRO_SETTINGS_JSON"; exit 0; fi
profile=$REPRO_PROFILE
printf 'Configuration: %s\n' "$config_file"
case "$profile" in
  ci)
    revision=22721f0acdfe129ba2a8937057dd321efecf8aa1
    image_digest=sha256:dc76851796905fa258a99edb0a330a7bbe2f7e74c2639888982b590beb35a830
    ;;
  original)
    revision=aa3fd1d9b7f5c7191dec11975ac0aa7c4c4a18c5
    image_digest=sha256:34f3bf8015bae3450de321d72e7de72ab4366ef496172ca3d65ce41e95de0632
    ;;
  *) printf 'Usage: %s ci|original\n' "$0" >&2; exit 2 ;;
esac
image="public.ecr.aws/q9t5s3a7/vllm-ci-test-repo@$image_digest"
source_repo="${REPRO_REPO_URL:-https://github.com/psv666/vllm-omni.git}"
gpu_ids="${GPUS:-0,1,2}"
if [[ ! "$gpu_ids" =~ ^[0-9]+,[0-9]+,[0-9]+$ ]]; then
  echo 'GPUS must contain exactly three physical GPU indices, e.g. 0,1,2.' >&2
  exit 2
fi
IFS=, read -r first_gpu second_gpu third_gpu <<< "$gpu_ids"
if [[ "$first_gpu" == "$second_gpu" || "$first_gpu" == "$third_gpu" || "$second_gpu" == "$third_gpu" ]]; then
  echo 'Choose three distinct GPUs.' >&2
  exit 2
fi
for command_name in git docker nvidia-smi tar; do
  command -v "$command_name" >/dev/null || { echo "Missing: $command_name" >&2; exit 2; }
done
for support_file in inside_container.sh repro_stderr.py; do
  [[ -f "$support_dir/$support_file" ]] || { echo "Missing $support_file" >&2; exit 2; }
done
# Ordinary local model folders are separate from the Hugging Face cache mount.
for model_var in QWEN3_OMNI_MODEL_DIR JUDGE_MODEL_DIR; do
  model_value="${!model_var:-}"
  if [[ -n "$model_value" ]]; then
    [[ -d "$model_value" ]] || { echo "$model_var is not a directory: $model_value" >&2; exit 2; }
    model_value="$(cd -- "$model_value" && pwd -P)"
    [[ -f "$model_value/config.json" ]] || { echo "$model_var has no config.json: $model_value" >&2; exit 2; }
    printf -v "$model_var" '%s' "$model_value"
  fi
done
if [[ -n "${DUPLEX_EVAL_DATASET_DIR:-}" ]]; then
  [[ -d "$DUPLEX_EVAL_DATASET_DIR" ]] || { echo "DUPLEX_EVAL_DATASET_DIR is not a directory: $DUPLEX_EVAL_DATASET_DIR" >&2; exit 2; }
  DUPLEX_EVAL_DATASET_DIR="$(cd -- "$DUPLEX_EVAL_DATASET_DIR" && pwd -P)"
fi
root_input="${REPRO_ROOT:-$HOME/issue9258-a100}"
mkdir -p -- "$root_input"
repro_root="$(cd -- "$root_input" && pwd -P)"
mkdir -p -- "$repro_root/snapshots" "$repro_root/runs" "${HF_CACHE_DIR:-$repro_root/hf-cache}"
cache_dir="$(cd -- "${HF_CACHE_DIR:-$repro_root/hf-cache}" && pwd -P)"
source_dir="$repro_root/snapshots/$revision"
run_dir="$(mktemp -d "$repro_root/runs/$profile-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
printf 'Profile: %s\nCommit: %s\nPhysical GPUs: %s\nSource: %s\nEvidence: %s\n' \
  "$profile" "$revision" "$gpu_ids" "$source_dir" "$run_dir"

if [[ ! -e "$source_dir" ]]; then
  # Publish only a completed checkout. An interrupted preparation leaves no
  # final snapshot for a later run to mistake for a usable source tree.
  staging_dir="$(mktemp -d "$repro_root/snapshots/.$revision.XXXXXX")"
  launcher_repo="$(git -C "$support_dir" rev-parse --show-toplevel 2>/dev/null || true)"
  if [[ -n "$launcher_repo" ]] &&
    ! git -C "$launcher_repo" config --get-regexp '^remote\..*\.promisor$' >/dev/null &&
    ! git -C "$launcher_repo" config --get extensions.partialClone >/dev/null &&
    git -C "$launcher_repo" cat-file -e "$revision^{commit}" 2>/dev/null; then
    echo 'Preparing pinned source from the existing local repository...'
    # A local clone owns its object files (hardlinked when possible); avoid
    # --shared, whose host-only alternates path would break inside Docker.
    git clone --local --no-checkout "$launcher_repo" "$staging_dir"
    git -C "$staging_dir" remote set-url origin "$source_repo"
  else
    if [[ "$REPRO_OFFLINE" == 1 ]]; then
      echo "Pinned source $revision is unavailable locally and offline is true; use a complete launcher checkout or set offline: false in YAML." >&2
      exit 2
    fi
    echo 'Fetching the pinned source in one shallow fetch...'
    git init "$staging_dir"
    git -C "$staging_dir" remote add origin "$source_repo"
    git -C "$staging_dir" -c gc.auto=0 fetch --depth=1 --no-tags --progress origin "$revision"
  fi
  git -C "$staging_dir" -c core.autocrlf=false checkout --detach "$revision"
  mv -T -- "$staging_dir" "$source_dir"
fi
[[ "$(git -C "$source_dir" rev-parse HEAD)" == "$revision" ]] || { echo 'Checkout SHA mismatch.' >&2; exit 2; }
git -C "$source_dir" diff --quiet || { echo 'Checkout has tracked changes; use another REPRO_ROOT.' >&2; exit 2; }
git -C "$source_dir" diff --cached --quiet || { echo 'Checkout has staged changes; use another REPRO_ROOT.' >&2; exit 2; }

printf '%s\n' "$revision" > "$run_dir/omni-commit.txt"
printf '%s\n' "$REPRO_SETTINGS_JSON" > "$run_dir/run-config.json"
printf '%s\n' "$image" > "$run_dir/image-reference.txt"
printf '%s\n' "$gpu_ids" > "$run_dir/physical-gpus.txt"
printf 'QWEN3_OMNI_MODEL_DIR=%s\nJUDGE_MODEL_DIR=%s\n' \
  "${QWEN3_OMNI_MODEL_DIR:-Hugging Face model ID}" "${JUDGE_MODEL_DIR:-Hugging Face model ID}" \
  > "$run_dir/local-model-host-paths.txt"
printf '%s\n' "${DUPLEX_EVAL_DATASET_DIR:-Hugging Face dataset ID}" > "$run_dir/local-dataset-host-path.txt"
nvidia-smi -q > "$run_dir/host-nvidia-smi.txt"
nvidia-smi topo -m > "$run_dir/host-gpu-topology.txt"
docker version > "$run_dir/docker-version.txt"
if docker image inspect "$image" >/dev/null 2>&1; then
  printf 'Using cached CI image: %s\n' "$image"
else
  if [[ "$REPRO_OFFLINE" == 1 ]]; then
    echo "CI image is not cached and offline is true: $image" >&2
    exit 2
  fi
  docker pull "$image"
fi
docker image inspect --format '{{.Id}} {{json .RepoDigests}}' "$image" > "$run_dir/image-digests.txt"

# NVIDIA Container Toolkit exposes only these three physical GPUs. Within the
# container, CUDA ordinals 0,1 run Omni and ordinal 2 runs the judge model.
# Keep the normal container PID namespace: test cleanup stays in this container.
docker_args=(run --rm --init --gpus "\"device=$gpu_ids\"" --ipc=host --network=host
  --entrypoint /bin/bash --workdir /opt/vllm-omni
  --mount "type=bind,src=$source_dir,dst=/opt/vllm-omni"
  --mount "type=bind,src=$run_dir,dst=/repro"
  --mount "type=bind,src=$cache_dir,dst=/hf-cache"
  --mount "type=bind,src=$support_dir,dst=/repro-support,readonly"
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e "CUDA_VISIBLE_DEVICES=0,1,2"
  -e HF_HOME=/hf-cache -e PYTHONUNBUFFERED=1
  -e HF_HUB_DOWNLOAD_TIMEOUT=300 -e HF_HUB_ETAG_TIMEOUT=60
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn
  -e "EXPECTED_OMNI_SHA=$revision" -e "REPRO_PROFILE=$profile"
  -e "PREFETCH=${PREFETCH:-1}" -e "CAPTURE_STDERR=${CAPTURE_STDERR:-1}")
if [[ -n "${QWEN3_OMNI_MODEL_DIR:-}" ]]; then
  docker_args+=(--mount "type=bind,src=$QWEN3_OMNI_MODEL_DIR,dst=/local-models/qwen3-omni,readonly"
    -e QWEN3_OMNI_MODEL_DIR=/local-models/qwen3-omni)
fi
if [[ -n "${JUDGE_MODEL_DIR:-}" ]]; then
  docker_args+=(--mount "type=bind,src=$JUDGE_MODEL_DIR,dst=/local-models/judge,readonly"
    -e JUDGE_MODEL_DIR=/local-models/judge)
fi
if [[ -n "${DUPLEX_EVAL_DATASET_DIR:-}" ]]; then
  docker_args+=(--mount "type=bind,src=$DUPLEX_EVAL_DATASET_DIR,dst=/local-datasets/omni-duplex-eval,readonly"
    -e DUPLEX_EVAL_DATASET_DIR=/local-datasets/omni-duplex-eval)
fi
# Forward optional network/auth settings by name; never dump token values.
for env_name in HF_ENDPOINT HF_TOKEN HF_HUB_OFFLINE HF_DATASETS_OFFLINE VLLM_LOGGING_LEVEL HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy; do
  if [[ -n "${!env_name:-}" ]]; then docker_args+=(-e "$env_name"); fi
done
set +e
docker "${docker_args[@]}" "$image" /repro-support/inside_container.sh 2>&1 | tee "$run_dir/container.log"
run_status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$run_status" > "$run_dir/container-exit-code.txt"
archive_tmp="$(mktemp "$repro_root/runs/.evidence-XXXXXX")"
# gzip creates its output asynchronously. Keep it outside the archived folder
# so creation cannot change that folder's metadata while tar is reading it.
if tar --exclude=evidence.tar.gz -czf "$archive_tmp" -C "$run_dir" .; then
  mv -- "$archive_tmp" "$run_dir/evidence.tar.gz"
  printf '\nExit code: %s\nEvidence archive: %s/evidence.tar.gz\n' "$run_status" "$run_dir"
else
  archive_status=$?
  printf 'Archive failed (exit %s); logs remain in %s; partial archive: %s\n' \
    "$archive_status" "$run_dir" "$archive_tmp" >&2
  if [[ "$run_status" -eq 0 ]]; then run_status=$archive_status; fi
fi
printf 'Full generated response media, if any: %s/tests/dfx/perf/results/\n' "$source_dir"
exit "$run_status"
