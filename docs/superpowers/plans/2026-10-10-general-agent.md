# General MASS agent implementation record

Updated 2026-10-10 after the user authorized rsikit changes first, then MASS.
This supersedes the initial MASS-only step wrapper and global trace-root plan.
[Design](../specs/2026-10-10-general-agent-design.md), [user guide](../../agent.md).

## Scope

Fixed-weight team execution and bounded workflow search on trusted Gymnasium adapters.
Preserve the paper pipeline and unrelated Bitcoin/LM Studio edits. No task-number
routing, model-generated Python, unrestricted host tools, automatic replay, weight
training, or claims of measured live-model performance gains.

## Runtime foundation — complete

- [x] rsikit stop reasons separate natural endings from runtime budget exhaustion.
- [x] Trusted public-feedback projection and Policy.context binding before reset.
- [x] Durable per-attempt events, Run step bounds, immutable evaluation identity,
      and retry_failed=False for failed/uncertain attempts.
- [x] Existing signatures and default scoring/retry behavior remain compatible.
- [x] Real-worker crash checks and independent review fixes for legacy caches,
      cleanup errors, and feedback-projector deadlines.
- [x] rsikit commit 5335e8c on codex/agent-runtime; supplied checkout on that branch.
      No push or merge. 65 focused tests pass; six full-suite failures match baseline.
      See rsikit docs/AGENT_RUNTIME_VALIDATION.md.

## MASS implementation — complete

- [x] Strict configuration and canonical data-only policy source in agent/schema.py.
- [x] TeamPolicy reset/act/observe/close lifecycle with separate calls for 1–4 roles.
- [x] Gym JSON conversion with validation before side effects, including composite
      integer-space coercion regressions, plus context/call/token/time bounds.
- [x] Explicit public feedback, no private evaluator info/artifacts/errors in prompts.
- [x] AsyncOpenAI transport reuses existing JSON/text helpers and leaves the paper
      backend unchanged. Replaces the proposed blocking-thread wrapper so cancellation
      cannot leave HTTP threads behind. Total deadlines include parent revision calls.
- [x] SequentialOptimizer baseline/initial/revision search, exact seed panels,
      finite scores, no partial promotion, strict mean improvement, tie retention.
- [x] Held-out evaluation follows selection; interrupted search preserves incumbent.
- [x] solve/search CLI, create-only output, trusted factory adapters, metadata/source
      digests, atomic summaries/checkpoints, and shared rsikit durable events.
- [x] Two task examples, config, requirements, lazy imports and setup documentation.
- [x] Tests consolidated in test_agent.py and test_agent_cli.py. Mandatory
      tools/test_agent.py rejects missing runtime or skipped integration checks.
- [x] Independent review fixes reproduced with failing tests, then verified green.
- [x] Manual agent CI accepts a full published rsikit commit. The local runtime commit
      still needs publication before hosted CI can fetch it; no hosted run claimed.
- [x] Final full MASS discovery, core-only checks, paper previews, manifest review,
      and whitespace/lint results recorded in [the validation note](../../agent-validation.md).

## Implementation adjustments

No custom MASS step wrapper, SQL event table, or global trace-root variable is needed:
rsikit now supplies those runtime contracts. Module files are mass/agent/{schema,
backend,policy,search,runtime,cli}.py; mass/__main__.py only dispatches the namespace.
Local tests use the existing rsikit interpreter without changing its dependencies;
the user guide provides a dedicated venv setup.

## Next boundary

Weight training requires a versioned trace-to-training adapter, disjoint split
manifests, model-serving provenance, and an actual completed model-update cycle.
This release provides runtime evidence and workflow optimization, not that cycle.
