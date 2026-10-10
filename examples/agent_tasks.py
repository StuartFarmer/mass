"""Resettable in-memory tasks; no host tools or external side effects."""

import gymnasium as gym


class ChoiceTask(gym.Env):
    observation_space = gym.spaces.Discrete(3)
    action_space = gym.spaces.Discrete(2)
    instructions = "Choose action 1."

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return 0, {}

    def step(self, action):
        return 1, float(action == 1), True, False, {"mass_feedback": "Choice recorded."}


class ToolTask(gym.Env):
    observation_space = gym.spaces.Discrete(3)
    action_space = gym.spaces.Dict(
        {"operation": gym.spaces.Discrete(2), "value": gym.spaces.Discrete(3)}
    )
    instructions = "Set the state to 2 using operation 0 and value 2; then finish using operation 1."

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = 0
        return self.state, {}

    def step(self, action):
        if action["operation"] == 0:
            self.state = int(action["value"])
            return self.state, 0.0, False, False, {"mass_feedback": "State updated."}
        return (
            self.state,
            float(self.state == 2),
            True,
            False,
            {"mass_feedback": "Finished."},
        )
