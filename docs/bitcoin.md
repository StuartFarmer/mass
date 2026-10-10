# Run the Bitcoin task

Task **61** adapts `../rsikit/rsikit/envs/bitcoin.py` to MASS's research-task
format. It preserves the daily observations, post-fee allocation targets,
$10,000 starting cash, 0.1% trading fees, terminal liquidation, and cash-profit
objective. The agent produces the simulator, policy, tests, report, and metrics.
The task definition itself is not a prebuilt trading strategy or Gym environment.

The [bare prompt](../tasks/bare/query61.txt) and
[initial workflow](../tasks/initial/query61_ourTeam.txt) use the existing MASS
runner. [configs/bitcoin.json](../configs/bitcoin.json) selects only task 61,
uses one workflow update, and writes to `runs/bitcoin`. It leaves the paper's
12-task configuration and splits intact.

## Preview locally

From the MASS repository root, without installing dependencies or a model:

```bash
python3 -m mass plan --config configs/bitcoin.json
python3 -m unittest discover -s tests -v
```

`plan` validates the configuration and prints commands; it does not execute
the task. The optional RHI integration test needs the core dependencies.

## Run with OpenRouter

Use [configs/bitcoin-openrouter.json](../configs/bitcoin-openrouter.json) for
GPT-OSS 120B Nitro. It uses one workflow update, proxy port 9002, and
`runs/bitcoin-openrouter` for outputs. Follow [OpenRouter setup](openrouter.md)
for dependencies and `OPENROUTER_API_KEY`, then run from this repository root:

```bash
source .venv-core/bin/activate
export PATH="$PWD/.runtime/node_modules/.bin:$PATH"
export MASS_BITCOIN_TRAIN_CSV="$(cd ../rsikit && pwd)/rsikit/envs/data/bitcoin_train.csv"
export EPISODE_CUDA_VISIBLE_DEVICES=""
python -m mass plan --config configs/bitcoin-openrouter.json
python -m mass rollout --config configs/bitcoin-openrouter.json --tasks 61 --count 1
```

`plan` is offline. `rollout` runs one paid research episode. To run the
three-episode workflow search and its optimizer/judge calls:

```bash
python -m mass search --config configs/bitcoin-openrouter.json --tasks 61
```

Use a fresh `run_dir` for retries after failed partial runs. OpenRouter supports
inference, collection, and ranking; it rejects rendering, training, and export.
The local model configurations below retain their separate training recipe.

## Set up and run search

For an installed local Qwen model, use the [LM Studio setup](lmstudio.md) and
`configs/bitcoin-lmstudio.json`. The following instructions use vLLM instead.

Follow [the installation and model-serving guide](running.md#install) for the
pinned Qwen runtime and vLLM server. Full local serving/training uses Linux and
NVIDIA GPUs. On a Mac, use it as the client and forward a compatible remote
server to `127.0.0.1:8001`; for example, `ssh -N -L 8001:127.0.0.1:8001 GPU_HOST`.
The client still needs Python 3.12, Node.js 22, Bash, jq, curl, git, and uv.
For search alone, the client needs only the core Python dependencies:

```bash
python3.12 -m venv .venv-core
source .venv-core/bin/activate
pip install -r requirements-core.txt
npm install --prefix .runtime @qwen-code/qwen-code@0.20.0
export PATH="$PWD/.runtime/node_modules/.bin:$PATH"
```

Supply the exact training CSV already present in the sibling rsikit checkout.
Run these commands from the MASS root in the same shell as the search:

```bash
export MASS_BITCOIN_TRAIN_CSV="$(cd ../rsikit && pwd)/rsikit/envs/data/bitcoin_train.csv"
test -r "$MASS_BITCOIN_TRAIN_CSV" || exit 1
export EPISODE_CUDA_VISIBLE_DEVICES=""
python -m mass search --config configs/bitcoin.json --tasks 61
```

No rsikit Python installation is needed. On another machine, copy just that
training CSV and set `MASS_BITCOIN_TRAIN_CSV` to its absolute path. The task
verifies SHA-256 `2825cfe512f0f1680453ce9db6b087449af80b45c048ddcf82b0e8c0e1109d21`.
The CSV contains 2,557 daily prices, 2016-09-19 through 2023-09-19. It is Coin
Metrics data under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/),
as recorded in rsikit's data README and `data/bitcoin/metadata.json`.

One search executes three task episodes: a bare reference, the initial agent
workflow, and one proposed workflow. Model judging compares their artifacts.
Outputs include:

- `runs/bitcoin/reference/query61/`: bare-task report, code, and results.
- `runs/bitcoin/workspaces/v0/query61_ourTeam/` and `v1/query61_ourTeam/`:
  initial and updated workflow artifacts.
- `runs/bitcoin/search/query61_ourTeam/retained_workflow.json`: selected workflow.
- `runs/bitcoin/logs/`: prompts, model requests, and run status.

For a single bare-task episode instead of workflow search, use:

```bash
python -m mass rollout --config configs/bitcoin.json --tasks 61 --count 1
```

Its outputs go to `runs/bitcoin/evaluation/workspaces/query61_r0/`. Change
`run_dir` in a copied config for a fresh experiment; saved runs reject changed
configurations and preserve partial episodes.

## Interpretation and later stages

MASS uses model judgments to select workflows; it does not automatically rank
them by simulator profit. The task requires numerical cash-profit metrics and
accounting checks as evidence. Inspect those artifacts before trusting a score.

The later rsikit validation interval, 2023-09-19 through 2026-09-18, is reserved
for evaluating the frozen policy after search. Do not expose it to task agents.
The shared boundary price creates no overlap in daily returns. This historical
holdout is distinct from MASS's SFT validation trajectory. The default runner
executes generated commands on the host with `--approval-mode yolo`; use an
isolated compute environment containing only the training CSV if the holdout
must be inaccessible. A prompt restriction alone does not enforce isolation.

Search does not update model weights. To continue through collection, ranking,
selection, rendering, training, and export, follow [the cycle guide](running.md)
with `--config configs/bitcoin.json`. Collection runs 18 candidate episodes only
if the retained workflow beats the reference. The configuration retains the
paper's costly GPU training settings; they are not tuned for a single task.
Export paths are `models/bitcoin-L1-bf16` and `models/bitcoin-L1-fp8`.
