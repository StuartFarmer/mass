"""Offline CLI and asynchronous provider contracts."""

import asyncio
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

if importlib.util.find_spec("rsikit") is None:
    raise unittest.SkipTest("requires the rsikit agent runtime")

from rsikit.evaluation import InfrastructureError, PolicyError

from mass.agent.backend import AgentBackend
from mass.agent.cli import execute
from mass.agent.policy import TeamPolicy
from mass.agent.schema import AgentConfig, ModelConfig, TaskConfig
from tests.test_agent import MODEL, TASK, ScriptedModel, workflow


class AgentCliTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_call_has_total_deadline(self):
        async def slow(**kwargs):
            await asyncio.sleep(0.03)
            return SimpleNamespace(choices=[], usage=None)

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=slow)),
            close=AsyncMock(),
        )
        with patch("mass.agent.backend.AsyncOpenAI", return_value=client):
            backend = AgentBackend(ModelConfig(**MODEL, timeout_seconds=0.01))
            with self.assertRaises(InfrastructureError):
                await backend.acall_json(system_prompt="JSON", user_prompt="x")
            await backend.close()

    async def test_interrupted_search_retains_measured_incumbent(self):
        config = AgentConfig(
            model=ModelConfig(**MODEL),
            workflow=workflow("team"),
            search={"revisions": 1},
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(
                TeamPolicy, "_make_backend", lambda _: ScriptedModel([{"action": 1}])
            ),
            patch(
                "mass.agent.cli.AgentBackend",
                return_value=ScriptedModel([InfrastructureError("offline")]),
            ),
        ):
            path = Path(directory) / "run"
            with self.assertRaises(InfrastructureError):
                await execute("search", TaskConfig(**TASK), config, path)
            self.assertTrue((path / "best.py").exists())
            self.assertEqual(
                json.loads((path / "summary.json").read_text())["status"], "interrupted"
            )
            self.assertIn(
                "revision_started", (path / "search" / "events.jsonl").read_text()
            )

    async def test_solve_and_search_use_workers_and_export_artifacts(self):
        config = AgentConfig(
            model=ModelConfig(**MODEL),
            workflow=workflow("initial"),
            search={"revisions": 0},
        )
        for mode in ("solve", "search"):
            with (
                tempfile.TemporaryDirectory() as directory,
                patch.object(
                    TeamPolicy,
                    "_make_backend",
                    lambda _: ScriptedModel([{"action": 1}]),
                ),
                patch("mass.agent.cli.AgentBackend", return_value=ScriptedModel([])),
            ):
                path = Path(directory) / mode
                result = await execute(mode, TaskConfig(**TASK), config, path, seed=3)
                self.assertEqual(result["status"], "completed")
                self.assertTrue((path / "best.py").exists())
                self.assertTrue((path / "best_workflow.json").exists())
                saved = json.loads((path / "summary.json").read_text())
                self.assertEqual(saved["status"], "completed")
                attempts = list((path / "attempts").rglob("episode.pkl"))
                self.assertEqual(len(attempts), 1 if mode == "solve" else 6)
                if mode == "search":
                    self.assertEqual(set(result["heldout"]), {"100", "101"})
                    prompts = (path / "search" / "events.jsonl").read_text()
                    self.assertNotIn('"seed": 100', prompts)
                with self.assertRaises(FileExistsError):
                    await execute(mode, TaskConfig(**TASK), config, path)

    async def test_exhaustion_is_not_reported_as_success(self):
        config = AgentConfig(
            model=ModelConfig(**MODEL), workflow=workflow(), limits={"max_steps": 1}
        )
        task = TaskConfig(**{**TASK, "environment": "examples.agent_tasks:ToolTask"})
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(
                TeamPolicy,
                "_make_backend",
                lambda _: ScriptedModel([{"action": {"operation": 0, "value": 2}}]),
            ),
        ):
            result = await execute("solve", task, config, Path(directory) / "run")
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["episodes"]["0"]["stop_reason"], "step_limit")

    async def test_provider_parses_json_and_keeps_usage_unknown_when_missing(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"action": 1}'))],
            usage=None,
            id="fixture",
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=AsyncMock(return_value=response))
            ),
            close=AsyncMock(),
        )
        with patch("mass.agent.backend.AsyncOpenAI", return_value=client):
            backend = AgentBackend(ModelConfig(**MODEL))
            self.assertEqual(
                await backend.acall_json(system_prompt="JSON", user_prompt="x", seed=7),
                {"action": 1},
            )
            self.assertIsNone(backend.last_metrics["usage"])
            self.assertEqual(client.chat.completions.create.call_args.kwargs["seed"], 7)
            response.choices[0].message.content = "invalid"
            with self.assertRaises(PolicyError):
                await backend.acall_json(system_prompt="JSON", user_prompt="x")
            response.choices[0].message.content = '{"action": NaN}'
            with self.assertRaises(PolicyError):
                await backend.acall_json(system_prompt="JSON", user_prompt="x")
            await backend.close()
            client.close.assert_awaited_once()

    async def test_lmstudio_fallback_and_responses_provider(self):
        import httpx
        from openai import BadRequestError

        rejected = BadRequestError(
            "'response_format.type' must be 'json_schema' or 'text'",
            response=httpx.Response(
                400, request=httpx.Request("POST", "http://localhost")
            ),
            body=None,
        )
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"action": 1}'))],
            usage=None,
            id="fixture",
        )
        create = AsyncMock(side_effect=[rejected, response])
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
            close=AsyncMock(),
            responses=SimpleNamespace(
                create=AsyncMock(
                    return_value=SimpleNamespace(
                        output_text='{"action": 0}', usage=None, id="fixture"
                    )
                )
            ),
        )
        with (
            patch("mass.agent.backend.AsyncOpenAI", return_value=client),
            patch.dict("os.environ", {"TEST_AGENT_KEY": "not-a-real-key"}),
        ):
            backend = AgentBackend(ModelConfig(**MODEL))
            self.assertEqual(
                await backend.acall_json(system_prompt="JSON", user_prompt="x"),
                {"action": 1},
            )
            self.assertNotIn("response_format", create.call_args.kwargs)
            await backend.close()
            backend = AgentBackend(
                ModelConfig(
                    backend="openai-responses",
                    model="fixture",
                    api_key_env="TEST_AGENT_KEY",
                )
            )
            self.assertEqual(
                await backend.acall_json(system_prompt="JSON", user_prompt="x"),
                {"action": 0},
            )
            await backend.close()


if __name__ == "__main__":
    unittest.main()
