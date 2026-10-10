# Agent validation — 2026-10-10

The new shared rsikit runtime was implemented first and committed as `5335e8c` on
`feat/agent-runtime`. MASS then adopted its stop reasons, public feedback projection,
durable evaluation context, step bounds, evaluation identity, and no-retry option.

## MASS checks

- Mandatory offline agent suite: 22 tests pass with no skips. This includes real
  rsikit workers on discrete and structured tasks, retained unfinished model-call
  evidence after a worker deadline, both CLI execution modes, interrupted search,
  strict promotion/ties, feedback privacy, provider mocks, and total call deadlines.
- Full working-tree discovery: 38 tests pass, no skips, using the existing rsikit
  interpreter without modifying its environment. This includes the user's existing
  LM Studio tests and paper-pipeline integration.
- Core-only Python without rsikit/OpenAI: 18 discovered cases, 11 pass and 7 skip
  because their optional runtime/provider dependencies are absent.
- Both `mass plan` previews (`paper.json` and `cycle2.json`) succeed in core-only
  Python; the agent CLI instead reports actionable setup instructions.
- Ruff and whitespace checks pass for new/changed agent code and tests.
- Source manifest includes the reviewed agent paths and hashes. The working tree
  still reports three pre-existing changes: README.md, harness_improvement/llm_backend.py,
  and tasks/README.md. The README's agent addition is included separately; unrelated
  content is not silently certified by refreshing all hashes.
- An isolated export of the staged handoff passes all 22 agent tests and the full
  125-file source-manifest check with zero problems. This verifies the implementation
  does not depend on the user's unrelated working-tree edits.

Tests were introduced before implementation. An independent implementation review
identified integer coercion inside composite Gym spaces and an absent whole-call
deadline for revision generation. Each issue received a failing regression followed
by a passing fix. Integer conversion validation covers Sequence, Graph, and OneOf
as well as arrays and natural Dict/Tuple actions. Network timeout and cancellation
checks use async transports; no blocking model thread is left behind.

## rsikit checks

65 focused runtime/evaluator/Run/optimizer/package tests pass. The clean baseline
full suite had two failures and four errors; the changed full suite reproduced the
same six failures, involving video, missing scientific dependencies, and terminal
output assertions. Twelve optional tests skipped. Three further review regressions
were verified in the focused run. See rsikit's `docs/AGENT_RUNTIME_VALIDATION.md` for
the exact failing tests. Its full suite is not green in this local environment.

## Not established

No live model calls, hosted CI, GPU training, performance improvement, hostile-code
isolation, or exactly-once external actions were tested. The new manual CI job needs
a published rsikit commit before it can run. Workflow search keeps weights fixed;
the separate training integration remains future work.

## macOS worker repair

The reported worker abort was traced to system proxy discovery after fork, before
any `model_started` event. An explicit HTTPX client with `trust_env=False` avoids
that initialization. A regression rejects proxy discovery during actual SDK client
construction, and another completes an actual SDK request inside an actual rsikit
worker against a local scripted HTTP server. The regression failed before the fix.
Full test discovery on the clean feature branch now passes 36 tests, no skips.
No live provider, user credentials, or paid model calls were used.
