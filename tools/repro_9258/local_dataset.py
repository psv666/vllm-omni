# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Expose one mounted Omni-DuplexEval snapshot with explicit split names."""

from __future__ import annotations

from pathlib import Path

import yaml


def select_snapshot(root: Path, revision: str | None) -> tuple[Path, str | None, str]:
    snapshots = root / "snapshots"
    if not snapshots.is_dir():
        selected_revision = root.name if root.parent.name == "snapshots" else None
        return root, selected_revision, "standalone directory"
    if revision:
        snapshot = snapshots / revision
        if not snapshot.is_dir():
            raise ValueError(
                f"The mounted dataset cache has no snapshot for requested revision {revision}: {snapshot}. "
                "Keep the complete cache root mounted; no download was attempted."
            )
        return snapshot.resolve(strict=True), revision, "requested revision"
    main_ref = root / "refs" / "main"
    if main_ref.is_file():
        revision = main_ref.read_text(encoding="utf-8").strip()
        if not revision or Path(revision).name != revision or revision in (".", ".."):
            raise ValueError(f"Invalid dataset cache ref: {main_ref}")
        snapshot = snapshots / revision
        if not snapshot.is_dir():
            raise ValueError(f"Dataset refs/main points to a missing snapshot: {snapshot}")
        return snapshot.resolve(strict=True), revision, "refs/main"
    if (snapshots / "main").is_dir():
        return (snapshots / "main").resolve(strict=True), None, "snapshots/main (revision unverified)"
    candidates = {path.resolve(strict=True) for path in snapshots.iterdir() if path.is_dir()}
    if len(candidates) != 1:
        raise ValueError(
            f"Dataset cache needs one snapshot or a readable refs/main; found {len(candidates)}: {snapshots}"
        )
    snapshot = candidates.pop()
    return snapshot, snapshot.name, "only cached snapshot"


def prepare_dataset(root: Path, destination: Path, *, revision: str | None, splits: list[str]) -> dict:
    """Write a dataset card referencing existing files, without copying media.

    Loading a cache root recursively merges snapshots and loses split identity.
    A per-run card names the requested splits explicitly, even when the cached
    snapshot has no README. Absolute paths retain access to the mounted blobs.
    """
    root = root.resolve(strict=True)
    snapshot, selected_revision, reason = select_snapshot(root, revision)
    if not snapshot.is_relative_to(root):
        raise ValueError(
            f"Dataset snapshot resolves outside the mounted cache: {snapshot}; mount the complete cache root"
        )
    data_files = []
    parquet_files = []
    for split in sorted(set(splits)):
        if not split.startswith(("PR_", "RTD_")) or Path(split).name != split:
            raise ValueError(f"Unsupported local Omni-DuplexEval split: {split!r}")
        files = sorted(snapshot.rglob(f"{split}-*-of-*.parquet"))
        if not files:
            raise ValueError(f"No Parquet shards for {split} in selected snapshot: {snapshot}")
        indices = []
        counts = set()
        for path in files:
            # Read in the container, where symlinks have the same visibility
            # as the subsequent preflight and benchmark processes.
            try:
                target = path.resolve(strict=True)
            except OSError as error:
                raise ValueError(
                    f"Unreadable dataset shard: {path}; mount the cache root containing blobs and snapshots"
                ) from error
            if not target.is_relative_to(root):
                raise ValueError(
                    f"Dataset shard links outside the mounted root: {path} -> {target}; mount the complete cache"
                )
            size = path.stat().st_size
            with path.open("rb") as stream:
                header = stream.read(4)
                stream.seek(max(0, size - 4))
                footer = stream.read(4)
            if size < 8 or header != b"PAR1" or footer != b"PAR1":
                raise ValueError(f"Dataset shard is not a complete Parquet file: {path}")
            _, index, _, count = path.stem.rsplit("-", 3)
            if not index.isdecimal() or not count.isdecimal():
                raise ValueError(f"Invalid dataset shard name: {path.name}")
            indices.append(int(index))
            counts.add(int(count))
            parquet_files.append({"path": path.relative_to(snapshot).as_posix(), "bytes": size})
        if len(counts) != 1 or counts != {len(files)} or sorted(indices) != list(range(len(files))):
            raise ValueError(f"Missing or duplicate {split} shards in {snapshot}: {[path.name for path in files]}")
        data_files.append({"split": split, "path": [path.as_posix() for path in files]})
    if not data_files:
        raise ValueError("No local dataset splits requested")
    destination.mkdir(parents=True, exist_ok=False)
    card = {"configs": [{"config_name": "default", "data_files": data_files}]}
    (destination / "README.md").write_text("---\n" + yaml.safe_dump(card, sort_keys=False) + "---\n", encoding="utf-8")
    return {
        "container_path": str(destination),
        "mounted_root": str(root),
        "snapshot_path": str(snapshot),
        "requested_revision": revision,
        "selected_revision": selected_revision,
        "selection_reason": reason,
        "revision_matches_requested": selected_revision == revision if revision and selected_revision else None,
        "splits": sorted(set(splits)),
        "parquet_files": parquet_files,
    }
