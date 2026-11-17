# Offline Silero VAD artifact

`silero_vad.onnx` is the unmodified artifact pinned by both reproduction source
revisions. It supplies the server-side voice activity detector required by the
`PR_correction` workload's `server_vad` turn detection.

- Source: [istupakov/silero-vad-onnx](https://huggingface.co/istupakov/silero-vad-onnx/blob/8b14476858ef240c50b3884bb38cc67290c1cc70/silero_vad.onnx)
- Revision: `8b14476858ef240c50b3884bb38cc67290c1cc70`
- SHA-256: `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`
- Size: 2,327,524 bytes
- Upstream: [Silero VAD](https://github.com/snakers4/silero-vad)
- License: MIT; see [LICENSE.silero-vad](LICENSE.silero-vad).

The reproduction branch carries this small model through Git so the target
server needs no Hugging Face download or `silero-vad` package installation.
The cached CI image must provide ONNX Runtime. Before starting Qwen, the runner
uses the pinned source's backend to verify the checksum, load the graph on CPU,
and score one silent frame. A missing runtime or incompatible graph stops the
run immediately. The Qwen engine and its first duplex session remain cold.
