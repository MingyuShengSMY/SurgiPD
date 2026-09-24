import gymnasium as gym
import numpy as np
from gymnasium.envs.box2d import BipedalWalker


class Reward:
    def __init__(self, *arg, **kwargs):
        self.prev_shaping = None
        return

    def get_reward(self, env: gym.Env | None, prev_obs, next_obs, raw_reward, act, step_n, terminated, truncated, env_rwd_fun=None):
        return raw_reward

    def reset(self):
        self.prev_shaping = None
