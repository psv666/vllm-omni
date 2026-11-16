#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Run on the Linux A100 host, from the directory containing all three files.
# Usage: GPUS=0,1,2 bash repro_9258.sh ci
#        GPUS=0,1,2 bash repro_9258.sh original
# Options: REPRO_ROOT, REPRO_REPO_URL, HF_CACHE_DIR, PREFETCH=0|1,
# CAPTURE_STDERR=0|1. The README also shows invocation from the repo root.
set -Eeuo pipefail

profile="${1:-ci}"
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
support_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
for support_file in inside_container.sh repro_stderr.py; do
  [[ -f "$support_dir/$support_file" ]] || { echo "Missing $support_file" >&2; exit 2; }
done
root_input="${REPRO_ROOT:-$HOME/issue9258-a100}"
mkdir -p -- "$root_input"
repro_root="$(cd -- "$root_input" && pwd -P)"
mkdir -p -- "$repro_root/repos" "$repro_root/runs" "${HF_CACHE_DIR:-$repro_root/hf-cache}"
cache_dir="$(cd -- "${HF_CACHE_DIR:-$repro_root/hf-cache}" && pwd -P)"
source_dir="$repro_root/repos/$revision"
run_dir="$(mktemp -d "$repro_root/runs/$profile-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
printf 'Profile: %s\nCommit: %s\nPhysical GPUs: %s\nSource: %s\nEvidence: %s\n' \
  "$profile" "$revision" "$gpu_ids" "$source_dir" "$run_dir"

if [[ ! -e "$source_dir" ]]; then
  git clone --filter=blob:none --no-checkout "$source_repo" "$source_dir"
  git -C "$source_dir" fetch origin "$revision"
  git -C "$source_dir" -c core.autocrlf=false checkout --detach "$revision"
fi
[[ "$(git -C "$source_dir" rev-parse HEAD)" == "$revision" ]] || { echo 'Checkout SHA mismatch.' >&2; exit 2; }
git -C "$source_dir" diff --quiet || { echo 'Checkout has tracked changes; use another REPRO_ROOT.' >&2; exit 2; }
git -C "$source_dir" diff --cached --quiet || { echo 'Checkout has staged changes; use another REPRO_ROOT.' >&2; exit 2; }

printf '%s\n' "$revision" > "$run_dir/omni-commit.txt"
printf '%s\n' "$image" > "$run_dir/image-reference.txt"
printf '%s\n' "$gpu_ids" > "$run_dir/physical-gpus.txt"
nvidia-smi -q > "$run_dir/host-nvidia-smi.txt"
nvidia-smi topo -m > "$run_dir/host-gpu-topology.txt"
docker version > "$run_dir/docker-version.txt"
docker pull "$image"
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
# Forward optional network/auth settings by name; never dump token values.
for env_name in HF_ENDPOINT HF_TOKEN VLLM_LOGGING_LEVEL HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy; do
  if [[ -n "${!env_name:-}" ]]; then docker_args+=(-e "$env_name"); fi
done
set +e
docker "${docker_args[@]}" "$image" /repro-support/inside_container.sh 2>&1 | tee "$run_dir/container.log"
run_status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$run_status" > "$run_dir/container-exit-code.txt"
tar --exclude=evidence.tar.gz -czf "$run_dir/evidence.tar.gz" -C "$run_dir" .
printf '\nExit code: %s\nEvidence archive: %s/evidence.tar.gz\n' "$run_status" "$run_dir"
printf 'Full generated response media, if any: %s/tests/dfx/perf/results/\n' "$source_dir"
exit "$run_status"
