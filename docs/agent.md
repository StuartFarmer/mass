# General MASS agent

MASS can execute a frozen team workflow and improve that workflow through measured
search on a trusted Gymnasium task adapter. The default team calls a proposer,
critic, and coordinator separately. Only the coordinator's validated action reaches
the environment. This release keeps model weights fixed; it does not implement
recursive weight training or convert these traces into the paper's training format.

## Install

Use Python 3.10+ and run from this repository root. rsikit is a source dependency:
the compatible implementation is commit `5335e8c` on its `feat/agent-runtime`
branch. An older package with the same `0.0.1` version is insufficient.

```sh
python3 -m venv .venv-agent
.venv-agent/bin/python -m pip install -e ~/Developer/AI/rsikit
.venv-agent/bin/python -m pip install -r requirements-agent.txt
.venv-agent/bin/python tools/test_agent.py
```

The final command requires the runtime and fails if integration tests are skipped.
Core-only test discovery may skip agent tests so the existing paper pipeline keeps
its smaller dependency set. A manual GitHub Actions agent job accepts a full,
published rsikit commit SHA as `rsikit_ref`. The new runtime commit must be published
before that hosted job can fetch it; ordinary core CI is unchanged.

## Run

Copy `configs/agent-example.json` to an ignored `configs/local-agent.json`. Set its
model ID and OpenAI-compatible local endpoint to match the model you serve. The
example works with vLLM or LM Studio; it does not launch a server or download weights.

```sh
.venv-agent/bin/python -m mass agent solve \
  --task tasks/agent-example.json --config configs/local-agent.json \
  --output runs/agent-solve --seed 0

.venv-agent/bin/python -m mass agent search \
  --task tasks/agent-example.json --config configs/local-agent.json \
  --output runs/agent-search
```

`solve` evaluates exactly the configured workflow and seed. `search` evaluates a
single-role baseline, the configured initial workflow, then at most `revisions`
model-proposed workflows. Each candidate receives the same selection seeds. Only a
complete, eligible panel can win, higher mean fitness wins, and ties keep the
incumbent. Malformed and duplicate revisions consume attempts. After selection,
held-out seeds evaluate the winner without feeding results into search.

The output directory must be new. `completed` means evaluation finished with eligible
episodes; it does not imply a particular task score or prove improvement. A failed
episode or all-failed search exits nonzero. An interrupted search retains the last
measured winner, but makes no claim that search completed. No execution is automatically
resumed or replayed.

For OpenAI Responses, use `backend: "openai-responses"`, an explicit model ID, and
`api_key_env: "OPENAI_API_KEY"`. Credentials are read from that environment variable;
never put their values in configuration, task kwargs, or instructions. Configuration
rejects unknown fields and URLs with embedded credentials, queries, or fragments.

## Add a task

A task JSON file specifies `environment: "your_package.tasks:YourFactory"`, an
optional `environment_kwargs` object, public `instructions`, and `task_version`.
The trusted factory returns a synchronous Gymnasium `Env`, with observation/action
spaces, `reset(seed=...)`, and `step(action)`. Change `task_version` when evaluator,
data, tool behavior, or hidden tests change. The canonical task configuration is
hashed into policy identity and the run's evaluation key.

The adapter owns tools, side effects, reset semantics, and grading. The same policy
supports numbers, arrays, strings, and structured actions through Gym's JSON codecs;
Dict and Tuple actions use natural objects and arrays. Integer values may not be
silently rounded. Custom spaces must provide valid reversible JSON codecs; normalize
unsupported representations in the adapter. `examples/agent_tasks.py` provides both
a discrete task and an in-memory state tool task. No task number or domain-specific
routing is needed in MASS itself.

`info['fitness']`, if supplied on the final step, is a finite numeric score to
maximize; otherwise rsikit sums rewards. Search requires resettable evaluations
without unreconciled external effects. Use isolated task state or task-specific
idempotency for external tools. Worker processes execute trusted Python; they are
not a hostile-code sandbox, and local traces do not provide exactly-once execution.

Only these fields may enter model feedback: public observations, actions, next
observations, rewards, completion flags, and the string `info['mass_feedback']`.
Adapters must deliberately publish that string and keep reference answers out of
all public fields. The full raw episode remains available locally for evaluation;
its other info keys, artifacts, and raw errors never enter prompts. Workflow
proposals see scalar selection scores, fixed stop categories, and bounded public
feedback. Held-out evidence stays outside proposal context.

## Limits and evidence

Defaults are 16 steps, 64 logical model calls per episode, 32768 UTF-8 request bytes,
a 300-second worker deadline, 2048 output tokens per call, and a 60-second total
model-call deadline. The call budget reserves a whole team before a step. Old
history is removed to fit the context limit; required task/current-step content
that cannot fit fails explicitly. Three workflow revision calls are allowed by
default; selection seeds are `[0,1]`, held-out seeds `[100,101]`.

SDK retries are disabled. A specific LM Studio response-format rejection permits
one compatibility retry within the same total deadline; successful metrics record
the transport request count. Logical-call limits are not exact HTTP request or
spending limits. Missing provider usage is recorded as unknown, not zero.

rsikit supplies the shared contracts:

- `Episode.stop_reason` separates natural termination, environment truncation,
  evaluator step exhaustion, policy budgets/errors, and worker timeouts. MASS
  excludes its step/call exhaustion from selection; natural completion or an
  environment limit on the last allowed step remains eligible. Generic rsikit
  scoring remains independent of an agent's selection policy.
- `Run.evaluate(max_steps=..., feedback=..., evaluation_key=..., retry_failed=False)`
  binds the evaluation contract and refuses automatic retries of uncertain or
  failed attempts. Feedback projection happens before `Policy.observe`; MASS
  additionally filters direct `observe` calls.
- `Policy.context` carries a per-attempt `EvaluationContext` before reset. Its
  durable append-only event sink does not affect policy identity or model context.
  Workers need no global trace-directory environment variable.

A run contains:

| Artifact | Meaning |
| --- | --- |
| `agent.json` | Task/config, agent version, MASS-agent and rsikit source digests |
| `summary.json` | Overall status, selected ID, evaluation/held-out outcomes |
| `search.json` | Candidate panels and promotions, checkpointed during search |
| `search/events.jsonl` | Incremental revision requests and outcomes |
| `attempts/<policy>/<seed>/<attempt>/events.jsonl` | Worker and model-call events, flushed durably as they happen |
| `attempts/.../episode.pkl`, `run.sqlite`, `exports/` | rsikit's factual episodes and policy storage |
| `best.py`, `best_workflow.json` | Measured winner, or the eligible fixed workflow in solve mode |

A started event without a finished event remains useful evidence after a crash.
It does not establish whether a remote request or external action completed. The
CLI creates a new run rather than guessing how to resume. Treat run data as private;
it includes task content and raw evaluator episodes.

`best.py` imports `mass.agent`; it needs compatible MASS and rsikit code plus the
configured model endpoint. It is a fixed workflow, not a standalone model or weights
file, and replay still makes model calls. Keep source digests and task/model versions
with comparisons: identical seeds do not make remote model behavior deterministic.

## Verification and future work

Run `tools/test_agent.py` for mandatory offline agent checks, and `python -m unittest
discover -s tests -v` for the full MASS suite. Tests include two task shapes through
real worker processes, promotion/ties, hidden-feedback sentinels, invalid action
coercion, model deadlines, interrupted search, and durable unfinished calls.

No live-model performance gain is established by these tests. Future weight training
needs a separate dataset export/validation and training adapter, model-version
identity, and fresh held-out evaluation. Durable traces and explicit visibility/stop
contracts support that extension without coupling rsikit to MASS's search algorithm.
