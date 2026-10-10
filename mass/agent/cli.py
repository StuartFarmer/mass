"""Create-only solve/search runs with durable rsikit worker evidence."""

import argparse
import asyncio
import hashlib
from pathlib import Path

import rsikit
from rsikit import EvaluationContext, Executor, Run, search
from rsikit.evaluation import InfrastructureError, PolicyError, episode_scores

from mass.io import read_json, write_json

from .backend import AgentBackend
from .runtime import load_environment
from .schema import AGENT_VERSION, AgentConfig, TaskConfig, policy_definition
from .search import WorkflowSearch, eligible


def source_digest(directory):
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*.py")):
        digest.update(
            str(path.relative_to(directory)).encode() + b"\0" + path.read_bytes()
        )
    return digest.hexdigest()


def panel_summary(episodes):
    scores = episode_scores(episodes)
    return {
        str(seed): {
            "score": scores.get(seed),
            "eligible": eligible(ep),
            "stop_reason": ep.stop_reason,
            "error": ep.error,
        }
        for seed, ep in episodes.items()
    }


async def execute(mode, task, config, output, *, seed=0):
    if mode not in {"solve", "search"} or type(seed) is not int or seed < 0:
        raise ValueError("Use solve/search with a nonnegative integer seed")
    # Refuse an existing output before task construction, which can itself own resources.
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    environment = load_environment(task)
    backend, optimizer = None, None
    summary = {
        "mode": mode,
        "status": "running",
        "task_fingerprint": task.fingerprint,
        "model_usage": "See per-call events; missing provider usage remains unknown.",
    }
    try:
        async with Run.create(name=f"mass-{mode}", path=output) as run:
            write_json(
                run.path / "agent.json",
                {
                    "task": task.model_dump(),
                    "config": config.model_dump(),
                    "agent_version": AGENT_VERSION,
                    "mass_agent_source": source_digest(Path(__file__).parent),
                    "rsikit_source": source_digest(Path(rsikit.__file__).parent),
                },
            )
            write_json(run.path / "summary.json", summary)
            events = EvaluationContext(
                policy_id="workflow-search",
                directory=run.path / "search",
                run_id=run.path.name,
            )

            def checkpoint(current):
                write_json(run.path / "search.json", current.result)
                if current.best:
                    current.best.to_file(run.path / "best.py")
                    write_json(
                        run.path / "best_workflow.json",
                        current.best_workflow.model_dump(),
                    )

            try:
                async with Executor(
                    episode_timeout=config.limits.episode_timeout
                ) as executor:

                    async def measure(policies, seeds):
                        return await run.evaluate(
                            policies,
                            environment=environment,
                            executor=executor,
                            seeds=seeds,
                            max_steps=config.limits.max_steps,
                            evaluation_key=f"mass-public-v1:{task.fingerprint}",
                            retry_failed=False,
                        )

                    if mode == "solve":
                        definition = policy_definition(
                            config.workflow,
                            config.model,
                            config.limits,
                            task_fingerprint=task.fingerprint,
                        )
                        result = await measure([definition], [seed])
                        summary["policy_id"] = definition.id
                        summary["episodes"] = panel_summary(result[definition.id])
                        success = all(
                            eligible(ep) for ep in result[definition.id].values()
                        )
                        if success:
                            definition.to_file(run.path / "best.py")
                            write_json(
                                run.path / "best_workflow.json",
                                config.workflow.model_dump(),
                            )
                    else:
                        backend = AgentBackend(config.model)
                        optimizer = WorkflowSearch(
                            task,
                            config.model,
                            config.limits,
                            config.search,
                            backend=backend,
                            initial_workflow=config.workflow,
                            on_event=events.emit,
                        )
                        best = await search(
                            optimizer,
                            lambda policies: measure(
                                policies, config.search.selection_seeds
                            ),
                            on_checkpoint=checkpoint,
                        )
                        summary["policy_id"] = best.id if best else None
                        summary["selection_mean"] = optimizer.best_mean
                        success = best is not None
                        if best:
                            heldout = (
                                await measure([best], config.search.heldout_seeds)
                            )[best.id]
                            summary["heldout"] = panel_summary(heldout)
                            success = all(eligible(ep) for ep in heldout.values())
                    summary["status"] = "completed" if success else "failed"
            except BaseException as exc:
                summary.update(status="interrupted", error_type=type(exc).__name__)
                raise
            finally:
                write_json(run.path / "summary.json", summary)
    finally:
        try:
            if optimizer is not None:
                await optimizer.aclose()
            if backend is not None:
                await backend.close()
        finally:
            environment.close()
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a MASS team on a trusted Gymnasium task adapter"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for mode in ("solve", "search"):
        command = subparsers.add_parser(mode)
        command.add_argument("--task", type=Path, required=True)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if mode == "solve":
            command.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        task = TaskConfig.model_validate(read_json(args.task))
        config = AgentConfig.model_validate(read_json(args.config))
        result = asyncio.run(
            execute(args.mode, task, config, args.output, seed=getattr(args, "seed", 0))
        )
    except (
        OSError,
        ValueError,
        TypeError,
        ImportError,
        AttributeError,
        InfrastructureError,
        PolicyError,
    ) as exc:
        parser.exit(2, f"Agent run failed: {exc}\n")
    print(f"{result['status']}: {args.output / 'summary.json'}")
    return 0 if result["status"] == "completed" else 1
