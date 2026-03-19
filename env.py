# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 13:30:29 2026

@author: Sourav
"""
import gymnasium as gym
import numpy as np

class AdversarialPendulum:
    def __init__(self, delta_max=0.1):
        self.env = gym.make("Pendulum-v1")
        self.delta_max = delta_max
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self):
        s, _ = self.env.reset()
        return s

    def step(self, a, a_bar):
        # step original env
        s_next, r, done, truncated, info = self.env.step(a)

        # adversary perturbs state
        a_bar = np.clip(a_bar, -self.delta_max, self.delta_max)

        s_next = s_next + a_bar

        return s_next, r, done, truncated, info