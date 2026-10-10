"""Offline contracts for the general team agent on rsikit."""

import asyncio
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

if importlib.util.find_spec("rsikit") is None:
    raise unittest.SkipTest("requires the rsikit agent runtime")

import gymnasium as gym
import numpy as np
from pydantic import ValidationError
from rsikit import Episode, Executor, Run, search
from rsikit.evaluation import InfrastructureError, PolicyBudgetExceeded, PolicyError

from mass.agent.policy import TeamPolicy, public_feedback, space_json
from mass.agent.schema import (
    AgentConfig,
    AgentLimits,
    ModelConfig,
    SearchConfig,
    TaskConfig,
    Workflow,
    policy_definition,
)
from mass.agent.search import WorkflowSearch

MODEL = {
    "backend": "local-vllm",
    "model": "offline",
    "base_url": "http://127.0.0.1:1234/v1",
}
TASK = {
    "instructions": "Choose action one.",
    "environment": "examples.agent_tasks:ChoiceTask",
    "task_version": "1",
}


def workflow(instruction="Choose a valid action"):
    return Workflow(roles=[{"id": "coordinator", "instruction": instruction}])


class ScriptedModel:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []
        self.closed = False

    async def acall_json(self, *, system_prompt, user_prompt, seed=None):
        self.requests.append((system_prompt, user_prompt))
        reply = next(self.replies)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    async def close(self):
        self.closed = True


class AgentSchemaTests(unittest.TestCase):
    def test_configuration_bounds_and_identity(self):
        for kwargs in (
            {"max_steps": 0},
            {"max_model_calls": True},
            {"episode_timeout": float("inf")},
        ):
            with self.assertRaises(ValidationError):
                AgentLimits(**kwargs)
        for kwargs in (
            {"selection_seeds": [True]},
            {"heldout_seeds": [0]},
            {"revisions": -1},
        ):
            with self.assertRaises(ValidationError):
                SearchConfig(**kwargs)
        for roles in (
            [],
            [{"id": "x", "instruction": "x"}] * 2,
            [{"id": str(i), "instruction": "x"} for i in range(5)],
        ):
            with self.assertRaises(ValidationError):
                Workflow(roles=roles)
        model, limits = ModelConfig(**MODEL), AgentLimits()
        first = policy_definition(workflow(), model, limits, task_fingerprint="task-v1")
        first.validate()
        self.assertEqual(
            first.id,
            policy_definition(workflow(), model, limits, task_fingerprint="task-v1").id,
        )
        self.assertNotEqual(
            first.id,
            policy_definition(workflow(), model, limits, task_fingerprint="task-v2").id,
        )
        self.assertNotEqual(
            first.id,
            policy_definition(
                workflow("new"), model, limits, task_fingerprint="task-v1"
            ).id,
        )
        with self.assertRaises(ValidationError):
            ModelConfig(**{**MODEL, "base_url": "http://user:secret@localhost/v1"})


class TeamPolicyTests(unittest.IsolatedAsyncioTestCase):
    def agent(self, backend, *, roles=None, limits=None, space=None):
        agent = TeamPolicy(
            gym.spaces.Discrete(3),
            space or gym.spaces.Discrete(2),
            instructions="Public task",
        )
        agent.CONFIG = {
            "workflow": (
                roles or AgentConfig(model=ModelConfig(**MODEL)).workflow
            ).model_dump(),
            "model": MODEL,
            "limits": (limits or AgentLimits()).model_dump(),
            "task_fingerprint": "fixture",
        }
        agent._make_backend = lambda: backend
        self.addAsyncCleanup(agent.close)
        return agent

    async def test_each_role_executes_and_only_coordinator_returns_action(self):
        model = ScriptedModel(
            [{"advice": "propose"}, {"advice": "check"}, {"action": 1}]
        )
        agent = self.agent(model)
        await agent.reset(seed=0)
        self.assertEqual(await agent.act(0), 1)
        self.assertEqual(len(model.requests), 3)
        self.assertIn("propose", model.requests[1][1])
        self.assertIn("check", model.requests[2][1])
        with self.assertRaises(RuntimeError):
            await agent.act(0)

    async def test_budget_and_oversized_context_fail_before_partial_team(self):
        for limits in (
            AgentLimits(max_model_calls=2),
            AgentLimits(max_context_bytes=1),
        ):
            model = ScriptedModel([])
            agent = self.agent(model, limits=limits)
            await agent.reset()
            with self.assertRaises((PolicyBudgetExceeded, PolicyError)):
                await agent.act(0)
            self.assertEqual(model.requests, [])

    async def test_invalid_action_and_response_are_candidate_failures(self):
        for reply in ({"action": 99}, {"action": 1, "unexpected": 0}, {"bad": 1}):
            agent = self.agent(ScriptedModel([reply]), roles=workflow())
            await agent.reset()
            with self.assertRaises(PolicyError):
                await agent.act(0)

    async def test_nested_spaces_and_integer_action_validation(self):
        space = gym.spaces.Dict(
            {
                "nested": gym.spaces.Tuple(
                    (
                        gym.spaces.Discrete(2),
                        gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32),
                    )
                )
            }
        )
        decoded = space_json(space, {"nested": [1, [0.5, -0.5]]}, decode=True)
        self.assertTrue(space.contains(decoded))
        self.assertEqual(space_json(space, decoded), {"nested": [1, [0.5, -0.5]]})
        agent = self.agent(
            ScriptedModel([{"action": [1.9, 0]}]),
            roles=workflow(),
            space=gym.spaces.MultiDiscrete([3, 3]),
        )
        await agent.reset()
        with self.assertRaises(PolicyError):
            await agent.act(0)

    async def test_composite_spaces_cannot_hide_integer_coercion(self):
        for space, raw in (
            (gym.spaces.Sequence(gym.spaces.Discrete(2)), [1.9]),
            (gym.spaces.Graph(gym.spaces.Discrete(2), None), {"nodes": [1.9]}),
            (gym.spaces.OneOf([gym.spaces.Discrete(2)]), [0, 1.9]),
        ):
            agent = self.agent(
                ScriptedModel([{"action": raw}]), roles=workflow(), space=space
            )
            await agent.reset()
            with self.assertRaises(PolicyError):
                await agent.act(0)

    async def test_history_is_bounded_and_finished_episode_needs_reset(self):
        from rsikit import Transition

        model = ScriptedModel([{"action": 1}, {"action": 1}])
        agent = self.agent(
            model, roles=workflow(), limits=AgentLimits(max_context_bytes=1000)
        )
        await agent.reset()
        await agent.act(0)
        await agent.observe(
            Transition(0, 1, 1, 1.0, False, False, {"mass_feedback": "x" * 3000})
        )
        await agent.act(1)
        self.assertNotIn("x" * 100, model.requests[-1][1])
        self.assertIn("Public task", model.requests[-1][1])
        await agent.observe(Transition(1, 1, 2, 1.0, True, False, {}))
        with self.assertRaises(RuntimeError):
            await agent.act(2)

    async def test_spaces_roundtrip_and_reset_clears_episode_state(self):
        space = gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        agent = self.agent(
            ScriptedModel([{"action": [0.5, -0.5]}]), roles=workflow(), space=space
        )
        await agent.reset()
        np.testing.assert_array_equal(
            await agent.act(0), np.array([0.5, -0.5], dtype=np.float32)
        )
        old = agent.backend
        agent._make_backend = lambda: ScriptedModel([{"action": [0, 0]}])
        await agent.reset(seed=2)
        self.assertTrue(old.closed)
        np.testing.assert_array_equal(await agent.act(0), [0, 0])

    async def test_private_info_never_enters_model_history(self):
        from rsikit import Transition

        model = ScriptedModel([{"action": 1}, {"action": 1}])
        agent = self.agent(model, roles=workflow())
        await agent.reset()
        await agent.act(0)
        await agent.observe(
            Transition(
                0,
                1,
                1,
                1.0,
                False,
                False,
                {"answer": "SECRET", "mass_feedback": "public"},
            )
        )
        await agent.act(1)
        self.assertNotIn("SECRET", model.requests[-1][1])
        self.assertIn("public", model.requests[-1][1])

    async def test_cancellation_and_infrastructure_are_not_bad_actions(self):
        for failure in (asyncio.CancelledError(), InfrastructureError("offline")):
            agent = self.agent(ScriptedModel([failure]), roles=workflow())
            await agent.reset()
            with self.assertRaises(type(failure)):
                await agent.act(0)


def episode(score, *, stop_reason="terminated", error=None):
    return Episode(
        observations=[0, 1],
        actions=[0],
        rewards=[score],
        terminations=[stop_reason == "terminated"],
        truncations=[stop_reason != "terminated"],
        infos=[{}, {"answer": "SECRET", "fitness": score}],
        stop_reason=stop_reason,
        error=error,
    )


class WorkflowSearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_duplicate_and_failed_proposals_are_bounded(self):
        initial = workflow("team")
        model = ScriptedModel([{"wrong": "shape"}, initial.model_dump()])
        agent = WorkflowSearch(
            TaskConfig(**TASK),
            ModelConfig(**MODEL),
            AgentLimits(),
            SearchConfig(revisions=2),
            backend=model,
            initial_workflow=initial,
        )
        count = 0

        async def measure(policies):
            nonlocal count
            count += 1
            return {
                policies[0].id: {seed: episode(100, error="SECRET") for seed in (0, 1)}
            }

        self.assertIsNone(await search(agent, measure))
        self.assertEqual(count, 2)
        self.assertEqual(agent.revision_calls, 2)
        self.assertNotIn("SECRET", json.dumps(model.requests))

    async def test_invalid_score_does_not_consume_pending_candidate(self):
        agent = WorkflowSearch(
            TaskConfig(**TASK),
            ModelConfig(**MODEL),
            AgentLimits(),
            SearchConfig(revisions=0),
            backend=ScriptedModel([]),
            initial_workflow=workflow("team"),
        )
        (candidate,) = await agent.propose()
        invalid = episode(1)
        invalid.infos[-1]["fitness"] = float("nan")
        for panel in (
            {candidate.id: {0: invalid, 1: episode(1)}},
            {"wrong": {0: episode(1), 1: episode(1)}},
        ):
            with self.assertRaises(ValueError):
                agent.update(panel)
            self.assertEqual(agent.evidence, [])
            self.assertIsNone(agent.best)
        agent.update({candidate.id: {0: episode(1), 1: episode(1)}})
        self.assertEqual(agent.best.id, candidate.id)

    async def test_measured_promotion_ties_and_private_feedback(self):
        model = ScriptedModel(
            [
                workflow("revision-one").model_dump(),
                workflow("revision-two").model_dump(),
            ]
        )
        agent = WorkflowSearch(
            TaskConfig(**TASK),
            ModelConfig(**MODEL),
            AgentLimits(),
            SearchConfig(revisions=2),
            backend=model,
            initial_workflow=workflow("initial"),
        )
        scores = iter([4, 3, 7, 7])
        evaluated = []

        async def measure(policies):
            score = next(scores)
            evaluated.append(policies[0])
            return {policies[0].id: {0: episode(score), 1: episode(score)}}

        best = await search(agent, measure)
        self.assertEqual(best.id, evaluated[2].id)
        self.assertTrue(agent.done)
        self.assertNotIn("SECRET", json.dumps(model.requests))

    async def test_partial_or_exhausted_panel_cannot_win(self):
        agent = WorkflowSearch(
            TaskConfig(**TASK),
            ModelConfig(**MODEL),
            AgentLimits(),
            SearchConfig(revisions=0),
            backend=ScriptedModel([]),
            initial_workflow=workflow("team"),
        )
        (first,) = await agent.propose()
        with self.assertRaises(ValueError):
            agent.update({first.id: {0: episode(100, error="failed")}})
        agent.update({first.id: {0: episode(2), 1: episode(2)}})
        (second,) = await agent.propose()
        agent.update(
            {second.id: {0: episode(100, stop_reason="step_limit"), 1: episode(100)}}
        )
        self.assertEqual((await search(agent, None)).id, first.id)


class WorkerIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_worker_deadline_keeps_unfinished_model_call(self):
        from examples.agent_tasks import ChoiceTask

        class SlowModel(ScriptedModel):
            async def acall_json(self, **kwargs):
                await asyncio.sleep(10)

        definition = policy_definition(
            workflow(), ModelConfig(**MODEL), AgentLimits(), task_fingerprint="timeout"
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(TeamPolicy, "_make_backend", lambda _: SlowModel([])),
        ):
            async with (
                Run.create(name="timeout", path=Path(directory) / "run") as run,
                Executor(episode_timeout=0.1) as executor,
            ):
                result = await run.evaluate(
                    [definition],
                    environment=ChoiceTask(),
                    executor=executor,
                    feedback=public_feedback,
                    evaluation_key="timeout",
                    max_steps=1,
                    retry_failed=False,
                )
                ep = result[definition.id][0]
                self.assertEqual(ep.stop_reason, "timeout")
                trace = next((run.path / "attempts").rglob("events.jsonl")).read_text()
                self.assertIn('"kind": "model_started"', trace)
                self.assertNotIn('"kind": "model_finished"', trace)

    async def test_real_workers_run_discrete_and_structured_tasks(self):
        from examples.agent_tasks import ChoiceTask, ToolTask

        for env, replies in (
            (ChoiceTask(), [{"action": 1}]),
            (
                ToolTask(),
                [
                    {"action": {"operation": 0, "value": 2}},
                    {"action": {"operation": 1, "value": 0}},
                ],
            ),
        ):
            definition = policy_definition(
                workflow(),
                ModelConfig(**MODEL),
                AgentLimits(),
                task_fingerprint="fixture",
            )
            with (
                tempfile.TemporaryDirectory() as directory,
                patch.object(
                    TeamPolicy,
                    "_make_backend",
                    lambda _, replies=replies: ScriptedModel(replies),
                ),
            ):
                async with (
                    Run.create(name="fixture", path=Path(directory) / "run") as run,
                    Executor() as executor,
                ):
                    result = await run.evaluate(
                        [definition],
                        environment=env,
                        executor=executor,
                        feedback=public_feedback,
                        evaluation_key="fixture",
                        max_steps=3,
                        retry_failed=False,
                    )
                    ep = result[definition.id][0]
                    self.assertIsNone(ep.error)
                    self.assertEqual(ep.total_reward, 1)
                    events = list((run.path / "attempts").rglob("events.jsonl"))
                    self.assertIn('"kind": "model_started"', events[0].read_text())


if __name__ == "__main__":
    unittest.main()
