#!/usr/bin/env python
# coding: utf-8
import gym
from gym import wrappers
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import deque
import numpy as np

from env import DroneEnv


env = DroneEnv()


# hyper parameters
EPISODES = 50  # number of episodes
EPS_START = 0.9  # e-greedy threshold start value
EPS_END = 0.05  # e-greedy threshold end value
EPS_DECAY = 200  # e-greedy threshold decay
GAMMA = 0.8  # Q-learning discount factor
LR = 0.001  # NN optimizer learning rate
BATCH_SIZE = 1  # Q-learning batch size


class DQNAgent:
    def __init__(self):
        self.model = nn.Sequential(
            nn.Linear(84, 21),
            nn.ReLU(),
            nn.Linear(21, 7)            
        )
        self.memory = deque(maxlen=10000)
        self.optimizer = optim.Adam(self.model.parameters(), LR)
        self.steps_done = 0
    
    def act(self, state):
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        print("eps_threshold: ", eps_threshold)
        self.steps_done += 1
        if random.random() > eps_threshold:
            print("case1 exploitation")
            #loss = orig_loss.max().detach().numpy()
            #print(self.model(state).data.max(1)[1])
            action = self.model(state).data.max(1)[1]
            action = [action.max(1)[1]]
            print("action: ", action)
            #action = self.model(state).data.max(1)[1].view(-1, 1)[1]
            #print(type(action))
            #print(action)
            return torch.LongTensor([action])
        else:
            print("case2 exploration")
            action = [random.randrange(0, 7)]
            print("action: ", action)
            return torch.LongTensor([action])

    def memorize(self, state, action, reward, next_state):
        self.memory.append((state,
                            action,
                            torch.FloatTensor([reward]),
                            torch.FloatTensor([next_state])))
    
    def learn(self):
        """Experience Replay"""
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states = zip(*batch)
        print(actions)
        states = torch.cat(states)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        next_states = torch.cat(next_states)
        print(actions)
        #current_q = self.model(states).gather(1, actions.unsqueeze((1)))
        current_q = self.model(states)
        max_next_q = self.model(next_states).detach().max(1)[0]
        expected_q = rewards + (GAMMA * max_next_q)
        
        loss = F.mse_loss(current_q.squeeze(), expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


agent = DQNAgent()
score_history = []
reward_history = []
score = 0

for e in range(1, EPISODES+1):
    state = env.reset()
    steps = 0
    while True:
        #env.render()
        state = torch.FloatTensor([state])
        action = agent.act(state)
        print(action)
        next_state, reward, done = env.step(action)

        # negative reward when attempt ends
        #if done:
        #    reward = -1

        agent.memorize(state, action, reward, next_state)
        agent.learn()

        state = next_state
        steps += 1
        score += reward
        
        if done:
            print("episode:{0}, reward: {1}, score: {2}".format(e, reward, score))
            print("----------------------------------------------------")
            score_history.append(steps)
            reward_history.append(reward)
            f = open('reward.txt', 'a')
            f.write(str(reward))
            f.close()
            f2 = open('score.txt', 'a')
            f2.write(str(score))
            f2.close()
            break
