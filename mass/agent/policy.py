"""A task-independent sequential team with explicit public feedback."""

from dataclasses import replace
from typing import ClassVar

from gymnasium import spaces
from pydantic import ValidationError
from rsikit import Policy
from rsikit.evaluation import PolicyBudgetExceeded, PolicyError

from .backend import AgentBackend
from .schema import Advice, AgentLimits, Decision, ModelConfig, Workflow, canonical


def public_feedback(transition):
    """Adapters explicitly publish feedback under mass_feedback; all other info is private."""
    value = transition.info.get("mass_feedback")
    info = {"mass_feedback": value} if isinstance(value, str) else {}
    return replace(transition, info=info)


def valid_json_conversion(original, encoded):
    """Reject structural changes and integer coercion; floating precision belongs to Gym."""
    if isinstance(encoded, dict):
        return (
            isinstance(original, dict)
            and original.keys() == encoded.keys()
            and all(
                valid_json_conversion(original[key], item)
                for key, item in encoded.items()
            )
        )
    if isinstance(encoded, (list, tuple)):
        return (
            isinstance(original, (list, tuple))
            and len(original) == len(encoded)
            and all(valid_json_conversion(a, b) for a, b in zip(original, encoded))
        )
    if type(encoded) in (int, float):
        return type(original) in (int, float) and (
            type(encoded) is float or original == encoded
        )
    return type(original) is type(encoded) and original == encoded


def space_json(space, value, *, decode=False):
    """Gym batches Dict/Tuple component-wise; expose one natural JSON action to models."""
    if isinstance(space, spaces.Dict):
        if not isinstance(value, dict) or set(value) != set(space.spaces):
            raise ValueError("Action keys must match the task space")
        return {
            key: space_json(child, value[key], decode=decode)
            for key, child in space.spaces.items()
        }
    if isinstance(space, spaces.Tuple):
        if not isinstance(value, (list, tuple)) or len(value) != len(space.spaces):
            raise ValueError("Action components must match the task space")
        values = [
            space_json(child, item, decode=decode)
            for child, item in zip(space.spaces, value)
        ]
        return tuple(values) if decode else values
    if not decode:
        return space.to_jsonable([value])[0]
    decoded = space.from_jsonable([value])[0]
    if not valid_json_conversion(value, space.to_jsonable([decoded])[0]):
        raise ValueError("Action structure and integer values must not be coerced")
    return decoded


class TeamPolicy(Policy):
    CONFIG: ClassVar[dict] = {}
    backend = None

    async def reset(self, *, seed=None):
        await self.close()
        await super().reset(seed=seed)
        self.workflow = Workflow.model_validate(self.CONFIG["workflow"])
        self.model = ModelConfig.model_validate(self.CONFIG["model"])
        self.limits = AgentLimits.model_validate(self.CONFIG["limits"])
        self.seed, self.calls = seed, 0
        self.history, self.pending, self.finished = [], None, False
        self.backend = self._make_backend()

    def _make_backend(self):
        return AgentBackend(self.model)

    def _emit(self, kind, data):
        if self.context is not None:
            self.context.emit(kind, data)

    def _observation(self, value):
        return space_json(self.observation_space, value)

    def _action(self, value):
        return space_json(self.action_space, value)

    def _prompt(self, system, payload):
        while True:
            user = canonical({**payload, "history": self.history})
            if (
                len(system.encode()) + len(user.encode())
                <= self.limits.max_context_bytes
            ):
                return user
            if not self.history:
                raise PolicyError("Required task context exceeds max_context_bytes")
            self.history.pop(0)

    async def act(self, observation):
        if self.pending is not None or self.finished:
            raise RuntimeError(
                "Observe the pending action, or reset a finished episode"
            )
        if self.calls + len(self.workflow.roles) > self.limits.max_model_calls:
            raise PolicyBudgetExceeded(
                "Insufficient model-call budget for the whole team"
            )
        current, advice = self._observation(observation), []
        for role in self.workflow.roles:
            coordinator = role.id == "coordinator"
            system = (
                'Choose an environment action. Return only a JSON object {"action": <valid action>}.'
                if coordinator
                else 'Advise the coordinator. Return only a JSON object {"advice": "concise advice"}.'
            )
            payload = {
                "task": self.instructions,
                "role": role.model_dump(),
                "observation": current,
                "action_space": str(self.action_space),
                "advice": advice,
            }
            user = self._prompt(system, payload)
            self.calls += 1
            self._emit(
                "model_started",
                {"call": self.calls, "role": role.id, "system": system, "prompt": user},
            )
            result = await self.backend.acall_json(
                system_prompt=system, user_prompt=user, seed=self.seed
            )
            self._emit(
                "model_finished",
                {
                    "call": self.calls,
                    "role": role.id,
                    "response": result,
                    "metrics": getattr(self.backend, "last_metrics", None),
                },
            )
            try:
                if coordinator:
                    raw = Decision.model_validate(result).action
                    action = space_json(self.action_space, raw, decode=True)
                    if not self.action_space.contains(action):
                        raise ValueError("Action outside the task's space")
                else:
                    advice.append(
                        {
                            "role": role.id,
                            "advice": Advice.model_validate(result).advice,
                        }
                    )
            except (
                ValidationError,
                ValueError,
                TypeError,
                KeyError,
                IndexError,
                OverflowError,
            ) as exc:
                raise PolicyError(
                    "Invalid role response or environment action"
                ) from exc
        self.pending = (current, self._action(action))
        return action

    async def observe(self, transition):
        transition = public_feedback(transition)
        observed = (
            self._observation(transition.observation),
            self._action(transition.action),
        )
        if self.pending is None or canonical(observed) != canonical(self.pending):
            raise RuntimeError("Feedback does not match the pending action")
        self.history.append(
            dict(
                observation=observed[0],
                action=observed[1],
                next_observation=self._observation(transition.next_observation),
                reward=transition.reward,
                terminated=transition.terminated,
                truncated=transition.truncated,
                **transition.info,
            )
        )
        self.pending = None
        self.finished = transition.terminated or transition.truncated

    async def close(self):
        if self.backend is not None:
            backend, self.backend = self.backend, None
            await backend.close()
