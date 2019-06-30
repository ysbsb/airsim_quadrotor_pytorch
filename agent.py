#!/usr/bin/env python
"""Environment for Microsoft AirSim Unity Quadrotor

- Author: Subin Yang
- Contact: subinlab.yang@gmail.com
"""
import math
import random
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import gym
from gym import wrappers
from env import DroneEnv

env = DroneEnv()

class DQN(nn.Module):
    def __init__(self, in_channels=84, num_actions=7):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 84, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(84, 42, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(42, 21, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(336, 168)
        self.fc5 = nn.Linear(168, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)


class Agent:
    def __init__(
        self,
        eps_start=0.9,
        eps_end=0.05,
        eps_decay=200,
        gamma=0.8,
        learning_rate=0.001,
        batch_size=1,
    ):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        dqn = DQN()
        self.model = dqn.forward()
        self.memory = deque(maxlen=10000)
        self.optimizer = optim.Adam(self.model.parameters(), self.learning_rate)
        self.steps_done = 0

    def act(self, state):
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.steps_done / self.eps_decay
        )
        self.steps_done += 1
        if random.random() > eps_threshold:
            action = self.model(state).data.max(1)[1]
            action = [action.max(1)[1]]
            return torch.LongTensor([action])
        else:
            action = [random.randrange(0, 7)]
            return torch.LongTensor([action])

    def memorize(self, state, action, reward, next_state):
        self.memory.append(
            (
                state,
                action,
                torch.FloatTensor([reward]),
                torch.FloatTensor([next_state]),
            )
        )

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)
        print(actions)
        states = torch.cat(states)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        next_states = torch.cat(next_states)
        print(actions)
        current_q = self.model(states)
        max_next_q = self.model(next_states).detach().max(1)[0]
        expected_q = rewards + (GAMMA * max_next_q)

        loss = F.mse_loss(current_q.squeeze(), expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        score_history = []
        reward_history = []
        score = 0        

        for e in range(1, EPISODES + 1):
        state = env.reset()
        steps = 0
        while True:
            state = torch.FloatTensor([state])
            action = act(state)
            print(action)
            next_state, reward, done = env.step(action)

            memorize(state, action, reward, next_state)
            learn()

            state = next_state
            steps += 1
            score += reward

            if done:
                print("episode:{0}, reward: {1}, score: {2}".format(e, reward, score))
                print("----------------------------------------------------")
                score_history.append(steps)
                reward_history.append(reward)
                f = open("reward.txt", "a")
                f.write(str(reward))
                f.close()
                f2 = open("score.txt", "a")
                f2.write(str(score))
                f2.close()
                break
