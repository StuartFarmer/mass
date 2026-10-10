"""Bounded workflow proposals selected by complete, comparable evaluations."""

from statistics import mean

from pydantic import ValidationError
from rsikit.evaluation import PolicyError, episode_scores
from rsikit.optimization import SequentialOptimizer, validate_results

from .schema import Workflow, canonical, policy_definition


def eligible(episode):
    return episode.error is None and episode.stop_reason in {
        "terminated",
        "environment_limit",
    }


class WorkflowSearch(SequentialOptimizer):
    def __init__(
        self, task, model, limits, config, *, backend, initial_workflow, on_event=None
    ):
        super().__init__()
        self.task, self.model, self.limits, self.config = task, model, limits, config
        self.backend, self.initial_workflow, self.on_event = (
            backend,
            initial_workflow,
            on_event,
        )
        self._best, self.best_workflow, self.best_mean = None, None, None
        self.attempts, self.workflows, self.revision_calls = [], {}, 0
        self._public_panel = []

    @property
    def best(self):
        return self._best

    @property
    def result(self):
        return {
            "attempts": self.attempts,
            "best_policy_id": self.best.id if self.best else None,
            "best_mean": self.best_mean,
            "revision_calls": self.revision_calls,
            "done": self.done,
        }

    def _event(self, kind, data):
        if self.on_event:
            self.on_event(kind, data)

    def _definition(self, workflow):
        definition = policy_definition(
            workflow, self.model, self.limits, task_fingerprint=self.task.fingerprint
        )
        if definition.id in self.workflows:
            self._event("duplicate_workflow", {"policy_id": definition.id})
            return None
        self.workflows[definition.id] = workflow
        return definition

    async def _proposals(self):
        baseline = Workflow(
            roles=[
                {
                    "id": "coordinator",
                    "instruction": "Choose the next valid action that best completes the task.",
                }
            ]
        )
        for workflow in (baseline, self.initial_workflow):
            definition = self._definition(workflow)
            if definition:
                yield definition
        for revision in range(self.config.revisions):
            system = (
                'Improve the workflow using measured feedback. Return only JSON {"roles": '
                '[{"id":"coordinator","instruction":"..."}]}. Use 1 to 4 unique lowercase role IDs; '
                "the last must be coordinator. Earlier roles give concise advice, the coordinator chooses actions."
            )
            payload = {
                "task": self.task.instructions,
                "incumbent": (self.best_workflow or self.initial_workflow).model_dump(),
                "incumbent_mean": self.best_mean,
                "latest_evaluation": self._public_panel,
            }
            user = canonical(payload)
            if (
                len(system.encode()) + len(user.encode())
                > self.limits.max_context_bytes
            ):
                raise PolicyError("Workflow revision context exceeds max_context_bytes")
            self.revision_calls += 1
            self._event(
                "revision_started",
                {"revision": revision, "system": system, "prompt": user},
            )
            try:
                response = await self.backend.acall_json(
                    system_prompt=system, user_prompt=user, seed=revision
                )
                self._event(
                    "revision_finished",
                    {
                        "revision": revision,
                        "response": response,
                        "metrics": getattr(self.backend, "last_metrics", None),
                    },
                )
                workflow = Workflow.model_validate(response)
                definition = self._definition(workflow)
            except (ValidationError, PolicyError, ValueError):
                self._event(
                    "revision_rejected",
                    {"revision": revision, "reason": "invalid_workflow"},
                )
                continue
            if definition:
                yield definition

    def update(self, results):
        pending = [self._pending_policy.id] if self._pending_policy else []
        validate_results(results, pending, seed_panel=set(self.config.selection_seeds))
        episodes = results[pending[0]]
        if set(episodes) != set(self.config.selection_seeds):
            raise ValueError(
                "Every candidate, including failures, needs the complete selection panel"
            )
        scores = episode_scores(episodes)
        public = []
        for seed, ep in episodes.items():
            # Only fixed categories, scalar scores and adapter-declared public text reach the proposer.
            feedback = ep.infos[-1].get("mass_feedback") if ep.infos else None
            public.append(
                {
                    "seed": seed,
                    "eligible": eligible(ep),
                    "stop_reason": ep.stop_reason,
                    "score": scores.get(seed) if eligible(ep) else None,
                    "feedback": feedback[:4096] if isinstance(feedback, str) else None,
                }
            )
        self._panel_eligible = all(eligible(ep) for ep in episodes.values())
        self._public_panel = public
        super().update(results)

    def _accept(self, scores, *, seed_scores, error):
        accepted = self._panel_eligible and error is None
        score = mean(scores) if accepted else None
        promoted = accepted and (self.best_mean is None or score > self.best_mean)
        if promoted:
            self._best, self.best_mean = self._pending_policy, score
            self.best_workflow = self.workflows[self.best.id]
        record = {
            "policy_id": self._pending_policy.id,
            "mean": score,
            "eligible": accepted,
            "promoted": promoted,
            "panel": self._public_panel,
        }
        self.attempts.append(record)
        self._event("candidate_evaluated", record)
