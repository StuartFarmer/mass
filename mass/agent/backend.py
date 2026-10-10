"""Async transport: cancellation never leaves a blocking model thread behind."""

import asyncio
import os

from openai import AsyncOpenAI, BadRequestError, DefaultAsyncHttpxClient, OpenAIError
from rsikit.evaluation import InfrastructureError, PolicyError

from harness_improvement.llm_backend import _extract_json_object, _responses_text

from .schema import canonical


class AgentBackend:
    def __init__(self, config):
        self.config = config
        key = os.environ.get(config.api_key_env)
        if not key and config.backend == "openai-responses":
            raise InfrastructureError(
                f"Set {config.api_key_env} for the selected model"
            )
        self.client = AsyncOpenAI(
            api_key=key or "EMPTY",
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            max_retries=0,
            # macOS system proxy discovery can abort a forked worker before any request.
            http_client=DefaultAsyncHttpxClient(
                trust_env=False, timeout=config.timeout_seconds
            ),
        )
        self.last_metrics = None

    async def acall_json(self, *, system_prompt, user_prompt, seed=None):
        self.last_metrics = None
        try:
            return await asyncio.wait_for(
                self._call_json(
                    system_prompt=system_prompt, user_prompt=user_prompt, seed=seed
                ),
                timeout=self.config.timeout_seconds,
            )
        except asyncio.TimeoutError:
            raise InfrastructureError(
                "Model call exceeded its total deadline"
            ) from None

    async def _call_json(self, *, system_prompt, user_prompt, seed=None):
        config = self.config
        requests = 1
        try:
            if config.backend == "openai-responses":
                response = await self.client.responses.create(
                    model=config.model,
                    instructions=system_prompt,
                    input=user_prompt,
                    text={"format": {"type": "json_object"}},
                    max_output_tokens=config.max_output_tokens,
                )
                raw = _responses_text(response)
            else:
                kwargs = {
                    "model": config.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_tokens": config.max_output_tokens,
                    "response_format": {"type": "json_object"},
                }
                if seed is not None:
                    kwargs["seed"] = seed
                try:
                    response = await self.client.chat.completions.create(**kwargs)
                except BadRequestError as exc:
                    if (
                        "'response_format.type' must be 'json_schema' or 'text'"
                        not in str(exc)
                    ):
                        raise
                    kwargs.pop("response_format")
                    requests += 1
                    response = await self.client.chat.completions.create(**kwargs)
                raw = response.choices[0].message.content if response.choices else ""
        except OpenAIError as exc:
            # Provider errors can echo request data or credentials; keep diagnostics local to the provider.
            raise InfrastructureError(
                f"Model transport failed ({type(exc).__name__})"
            ) from None
        usage = getattr(response, "usage", None)
        self.last_metrics = {
            "usage": usage.model_dump() if usage is not None else None,
            "response_id": getattr(response, "id", None),
            "transport_requests": requests,
        }
        try:
            payload = _extract_json_object(raw or "")
            canonical(payload)
            return payload
        except (ValueError, RuntimeError) as exc:
            raise PolicyError("Model response was not a JSON object") from exc

    async def close(self):
        await self.client.close()
