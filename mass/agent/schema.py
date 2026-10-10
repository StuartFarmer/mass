"""Validated, data-only workflows and reproducible policy definitions."""

import hashlib
import json
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator
from rsikit import PolicyDefinition

AGENT_VERSION = 1
Nonblank = Annotated[str, Field(pattern=r"\S", max_length=8192)]
Seed = Annotated[int, Field(ge=0)]


class StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)


class Role(StrictModel):
    id: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")]
    instruction: Nonblank


class Workflow(StrictModel):
    roles: list[Role] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def valid_roles(self):
        if len({role.id for role in self.roles}) != len(self.roles):
            raise ValueError("Role IDs must be unique")
        if self.roles[-1].id != "coordinator":
            raise ValueError("The last role must be coordinator")
        return self


class ModelConfig(StrictModel):
    backend: Literal["local-vllm", "openai-responses"]
    model: Nonblank
    base_url: str | None = None
    api_key_env: str = Field(
        default="OPENAI_API_KEY", pattern=r"^[A-Za-z_][A-Za-z0-9_]*$"
    )
    max_output_tokens: int = Field(default=2048, gt=0)
    timeout_seconds: float = Field(default=60.0, gt=0)

    @model_validator(mode="after")
    def valid_endpoint(self):
        if self.backend == "local-vllm" and not self.base_url:
            raise ValueError("local-vllm requires base_url")
        if self.base_url:
            url = urlsplit(self.base_url)
            if (
                url.scheme not in {"http", "https"}
                or not url.hostname
                or url.username
                or url.password
                or url.query
                or url.fragment
            ):
                raise ValueError(
                    "base_url must be HTTP(S), without credentials, query, or fragment"
                )
        return self


class AgentLimits(StrictModel):
    max_steps: int = Field(default=16, gt=0)
    max_model_calls: int = Field(default=64, gt=0)
    max_context_bytes: int = Field(default=32768, gt=0)
    episode_timeout: float = Field(default=300.0, gt=0)


class SearchConfig(StrictModel):
    revisions: int = Field(default=3, ge=0)
    selection_seeds: list[Seed] = Field(default_factory=lambda: [0, 1], min_length=1)
    heldout_seeds: list[Seed] = Field(default_factory=lambda: [100, 101], min_length=1)

    @model_validator(mode="after")
    def valid_panels(self):
        for panel in (self.selection_seeds, self.heldout_seeds):
            if len(panel) != len(set(panel)):
                raise ValueError("Seed panels must not contain duplicates")
        if set(self.selection_seeds) & set(self.heldout_seeds):
            raise ValueError("Selection and held-out seeds must be disjoint")
        return self


class TaskConfig(StrictModel):
    instructions: Nonblank
    environment: str = Field(pattern=r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*$")
    environment_kwargs: dict[str, Any] = Field(default_factory=dict)
    task_version: Nonblank

    @property
    def fingerprint(self):
        return hashlib.sha256(canonical(self.model_dump()).encode()).hexdigest()


def default_workflow():
    return Workflow(
        roles=[
            Role(
                id="proposer",
                instruction="Suggest a valid action that advances the task. Give concise actionable advice.",
            ),
            Role(
                id="critic",
                instruction="Check the suggestion against the task and observed feedback. Give concise corrections.",
            ),
            Role(
                id="coordinator",
                instruction="Use the advice and observed evidence to choose the next valid action.",
            ),
        ]
    )


class AgentConfig(StrictModel):
    model: ModelConfig
    workflow: Workflow = Field(default_factory=default_workflow)
    limits: AgentLimits = Field(default_factory=AgentLimits)
    search: SearchConfig = Field(default_factory=SearchConfig)


class Advice(StrictModel):
    advice: Nonblank


class Decision(StrictModel):
    action: Any


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def policy_definition(workflow, model, limits, *, task_fingerprint):
    config = canonical(
        {
            "workflow": workflow.model_dump(),
            "model": model.model_dump(),
            "limits": limits.model_dump(),
            "task_fingerprint": task_fingerprint,
            "agent_version": AGENT_VERSION,
        }
    )
    source = (
        "import json\nfrom mass.agent.policy import TeamPolicy\n\n"
        f"class Solution(TeamPolicy):\n    CONFIG = json.loads({config!r})\n"
    )
    if len(source.encode()) > 65536:
        raise ValueError("Rendered policy exceeds rsikit's 64 KiB source limit")
    return PolicyDefinition(source=source, name="MASS team")
