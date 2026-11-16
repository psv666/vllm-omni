# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Optional pytest diagnostics: tee the test server's piped stderr to files.

The fixture still reads the same stream. No RPC, timeout, or model setting changes.
Only subprocesses launched as a vLLM/vLLM-Omni module with stderr=PIPE are wrapped.
"""

import json
import os
import subprocess
from pathlib import Path


class _TeeStderr:
    def __init__(self, stream, path):
        self._stream = stream
        self._path = path

    def __iter__(self):
        with self._path.open("ab") as destination:
            for line in self._stream:
                destination.write(line.encode("utf-8", errors="replace") if isinstance(line, str) else line)
                destination.flush()
                yield line

    def __getattr__(self, name):
        return getattr(self._stream, name)


def pytest_configure(config):
    original = subprocess.Popen
    destination = Path(os.environ.get("REPRO_STDERR_DIR", "/repro/server-stderr"))
    destination.mkdir(parents=True, exist_ok=True)

    class CapturedPopen(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            argv = self.args
            is_server = isinstance(argv, (list, tuple)) and any(
                isinstance(part, str) and part.startswith(("vllm_omni.entrypoints.", "vllm.entrypoints."))
                for part in argv
            )
            if is_server and self.stderr is not None and kwargs.get("stderr") == subprocess.PIPE:
                log = destination / f"server-{self.pid}.stderr.log"
                (destination / f"server-{self.pid}.command.json").write_text(json.dumps(argv, indent=2) + "\n")
                self.stderr = _TeeStderr(self.stderr, log)

    subprocess.Popen = CapturedPopen
