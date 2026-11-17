#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Collect complete diagnostic text from a finished run, without rerunning it.
set -Eeuo pipefail

if [[ "$#" -ne 1 || ! -d "$1" ]]; then
  echo "Usage: $0 /absolute/path/to/finished/run" >&2
  exit 2
fi
run_dir="$(cd -- "$1" && pwd -P)"
if [[ ! -f "$run_dir/container-exit-code.txt" || ! -f "$run_dir/container.log" ]]; then
  echo "Run has not finished or has no container log: $run_dir" >&2
  exit 2
fi
output_tmp="$(mktemp "$run_dir/.diagnostics-XXXXXX")"
trap 'rm -f -- "$output_tmp"' EXIT
{
  printf 'Issue #9258 diagnostic text\nRun directory: %s\n' "$run_dir"
  for source in \
    "$run_dir/container-exit-code.txt" "$run_dir/inside-exit-code.txt" "$run_dir/pytest-exit-code.txt" \
    "$run_dir/run-config.json" "$run_dir/summary.json" \
    "$run_dir"/env/*.json "$run_dir"/env/*.log "$run_dir"/env/*.txt "$run_dir"/env/*.yaml \
    "$run_dir"/server-stderr/*.command.json "$run_dir"/server-stderr/*.stderr.log \
    "$run_dir"/results/result_*.json "$run_dir/container.log"; do
    [[ -f "$source" ]] || continue
    printf '\n===== %s =====\n' "${source#"$run_dir"/}"
    cat -- "$source"
    printf '\n'
  done
} > "$output_tmp"
mv -- "$output_tmp" "$run_dir/diagnostics.txt"
printf 'Text diagnostics: %s/diagnostics.txt\n' "$run_dir"
