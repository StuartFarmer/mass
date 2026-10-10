# Run MASS with LM Studio

The [Bitcoin LM Studio configuration](../configs/bitcoin-lmstudio.json) uses
`qwen/qwen3.6-27b` at `http://127.0.0.1:1234/v1`, the logging proxy on port 9001,
and `runs/bitcoin-lmstudio` for outputs. It uses the existing local
OpenAI-compatible backend; the internal backend name remains `local-vllm`.

This setup uses the installed 4-bit MLX model, rather than the paper's FP8
checkpoint. It is for local inference and workflow search. The configuration's
training fields retain the paper recipe and do not train or export the loaded
LM Studio model.

## Environment

From the repository root, install the client dependencies and pinned coding
runtime if they are not already present:

```bash
uv venv --python 3.12 .venv-core
uv pip install --python .venv-core/bin/python -r requirements-core.txt
npm install --prefix .runtime @qwen-code/qwen-code@0.20.0
```

For each new shell:

```bash
source .venv-core/bin/activate
export PATH="$PWD/.runtime/node_modules/.bin:$PATH"
export EPISODE_CUDA_VISIBLE_DEVICES=""
export MASS_BITCOIN_TRAIN_CSV="$(cd ../rsikit && pwd)/rsikit/envs/data/bitcoin_train.csv"
```

## Model server

Start LM Studio's local server and load the installed model with the context
length used by MASS. If it is already loaded with these settings, reuse it:

```bash
lms server start --port 1234
lms load qwen/qwen3.6-27b --identifier qwen/qwen3.6-27b --context-length 262144 --yes
lms ps --json
curl --fail http://127.0.0.1:1234/v1/models
```

The loaded model must report `contextLength: 262144`, and `/v1/models` must
include `qwen/qwen3.6-27b`. Keep server authentication disabled for this local
setup, and bind to loopback. MASS defaults to the placeholder key `EMPTY`
locally; authenticated remote serving is covered by the
[OpenRouter guide](openrouter.md).

LM Studio exposes [OpenAI-compatible APIs](https://lmstudio.ai/docs/developer/openai-compat)
and [tool calling](https://lmstudio.ai/docs/developer/openai-compat/tools).
For its explicit rejection of `response_format.type=json_object`, MASS retries
once without constrained decoding and still parses and validates the returned
JSON. Other failures are not retried. This avoids the empty final content
observed with constrained JSON-schema output on this Qwen/MLX installation.

## Run

Preview without model calls:

```bash
python -m mass plan --config configs/bitcoin-lmstudio.json
```

Run one bare Bitcoin task, or run the three-episode workflow search:

```bash
python -m mass rollout --config configs/bitcoin-lmstudio.json --tasks 61 --count 1
python -m mass search --config configs/bitcoin-lmstudio.json --tasks 61
```

These are separate operations; use the one you want. Search executes the bare
reference, initial workflow, and one proposed workflow plus model judgments.
See [the Bitcoin guide](bitcoin.md) for artifacts, scoring, and the reserved
historical holdout. Use a fresh `run_dir` in a copied config when changing model
settings or retrying a failed partial run.

Generated task commands run on the host in Qwen Code's YOLO mode. A local model
does not sandbox those commands. Long research episodes and large prompt
prefills can take substantial time on a Mac; a short compatibility test does
not establish full-task runtime or result quality.

## Setup verification (2026-10-10)

Verified the loaded 262,144-token context, Qwen Code 0.20.0, and the actual MASS
JSON and text backends against the local model. A complete episode made three
chat requests, wrote and read back `MASS_LMSTUDIO_OK`, and recorded successful
status and compressed API logs under `runs/lmstudio-smoke/`. All 16 offline
tests passed with the core dependencies installed. The full Bitcoin task,
weight training, and OpenRouter execution were not run.
