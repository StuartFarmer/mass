"""Trusted task adapters; capabilities and reset semantics belong to the task."""

import importlib

import gymnasium as gym


def load_environment(task):
    module, name = task.environment.split(":")
    factory = getattr(importlib.import_module(module), name)
    environment = factory(**task.environment_kwargs)
    if not isinstance(environment, gym.Env):
        raise TypeError("Task factory must return a Gymnasium Env")
    wrapped = gym.Wrapper(environment)
    wrapped.instructions = task.instructions
    return wrapped
