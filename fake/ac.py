# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by Mengqi.Ye on 2020/12/21
"""

from collections import namedtuple
from enum import Enum

from a9c_gym.envs import FakeEnv
from torch import nn, Tensor

from gym import Env
from gym.spaces import Space, Box
import a9c_gym


def exam_box_space(space: Space, name=''):
    print(f"{name} Space details:")
    print(f"{name} shape : {space.shape}")
    print(f"{name} high : {space.high}")
    print(f"{name} low : {space.low}")


def exam_env_space(env: Env):
    exam_box_space(env.action_space, name='Fake Env Action Space')
    exam_box_space(env.observation_space, name='Fake Env Observation Space')


ACTION_MEANING = {
    0: "mean",
    1: "std",
}

Trajectory = namedtuple(
    'Trajectory',
    ['env_id', 'frame_count', 'fps_avg', 'reward', 'observation', 'next_observation', 'state', 'next_state', 'action']
)


class ActorStrategy(Enum):
    BASE = 1,
    MaximizeSth = 1 << 4,
    MaximizeEntropy = (1 << 4) + 1,
    MinimizeSth = 2 << 4,


class FakeAgentA3C(nn.Module):
    def __init__(self, feature_size=100):

        linear_features_list = []
        while feature_size > 1:
            feature_size_smaller = feature_size // 2
            linear_features_list.append((feature_size, feature_size_smaller))
            feature_size = feature_size_smaller

        self.strategies = nn.ModuleDict({
            'linear': nn.ModuleList([nn.Linear(*t) for t in linear_features_list]),
        })

        self.activation = nn.ModuleDict({
            'lrelu': nn.LeakyReLU(),
            'prelu': nn.PReLU()
        })

        self.agent_states: Tensor = None
        self.env_states: Tensor = None

    def forward(self, env: FakeEnv):
        obs = env.observation
        if obs == None:
            action = env.action_space.sample()
        else:
            action = self.strategy['linear'](obs)
        env.step(action)

    def register_strategy(self, s: ActorStrategy, m: nn.Module):
        if s in self.stagegies:
            raise Warning(f'{s} is already registered.')

        else:
            self.strategies[s] = m

    def unregister_strategy(self, s: ActorStrategy):
        if s in self.strategies:
            self.strategies.__delitem__(s)
        else:
            raise Warning(f'{s} not in registered strategies')

    def observe(self, env: FakeEnv) -> Tensor:
        """
        Returns the encoded state of the environment
        """
        self.env_states = env.observation

    def act(self, env: FakeEnv, s: ActorStrategy):
        """
        Do something to the env.
        """
        action = env.action_space.sample()
        print(f"Sampled action shape : {action.shape}")
        env.step(action)


Critique = namedtuple('state', 'critic_value_dict')


class CriticStrategy(Enum):
    BASE = 1,
    MaximizeSth = 1 << 4,
    MaximizeEntropy = (1 << 4) + 1,
    MinimizeSth = 2 << 4,


class FakeCriticA3C(nn.Module):
    def __init__(self):
        self.strategies = nn.ModuleDict({

        })

    def register_strategy(self, s: CriticStrategy, m: nn.Module):
        if s in self.modules():
            raise Warning(f'{s} is already registered.')
        else:
            self.stagegies[s] = m

    def unregister_strategy(self, s: CriticStrategy):
        if s in self.stagegies:
            self.stagegies.__delitem__(s)
        else:
            raise Warning(f'{s} not in registered strategies')

    def criticize(self, env: FakeEnv) -> Tensor:
        """
        Returns the encoded state of the environment
        """
        c = Critique(env.observation)


class FakeActorCritic(nn.Module):
    def __init__(self,
                 observation_space: Box, action_space: Box,
                 hidden_sizes=(256, 256), activation=nn.ReLU, device='cpu'
                 ):
        super(FakeActorCritic, self).__init__()
        self.observation_space = observation_space
        self.action_spaced = action_space

        exam_box_space(action_space)

        self.actor = FakeAgentA3C()
        self.critic = FakeCriticA3C()

    def forward(self, x):
        policy = self.actor(x)
