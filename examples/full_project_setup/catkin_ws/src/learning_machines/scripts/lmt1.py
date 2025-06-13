import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from robobo_interface import SimulationRobobo, HardwareRobobo

# Hyperparameters
LEARNING_RATE = 0.001
GAMMA = 0.99
BATCH_SIZE = 64
REPLAY_MEMORY_SIZE = 10000
TARGET_UPDATE_FREQ = 10

NUM_ACTIONS = 3  # forward, left, right
OBSERVATION_SPACE = 5 

# IR thresholds 
THRESHOLD_COLLISION = [160, 170, 180, 170, 160]
THRESHOLD_NEAR =      [90, 95, 100, 95, 90]

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(OBSERVATION_SPACE, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, NUM_ACTIONS)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ObstacleAvoidanceAgent:
    def __init__(self):
        self.policy_net = DQN()
        self.target_net = DQN()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.steps_done = 0

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, NUM_ACTIONS - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                return self.policy_net(state_tensor).argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states)
        next_states = np.array(next_states)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.policy_net(states).gather(1, actions).squeeze()
        
        next_q_values = self.target_net(next_states).max(1)[0]
        expected_q_values = rewards + GAMMA * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())

def run_training_episode(rob: SimulationRobobo, agent: ObstacleAvoidanceAgent, epsilon: float):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    total_reward = 0.0
    done = False
    max_steps = 100
    step_count = 0
    prev_wheels = rob.read_wheels()

    while not done and step_count < max_steps:
        step_count += 1
        
        irs = rob.read_irs()
        state_raw = [irs[7], irs[3], irs[4], irs[5], irs[6]]
        state = np.array(state_raw) / 200.0

        if isinstance(rob, HardwareRobobo):
            state = state * 1.5

        action = agent.select_action(state, epsilon)

        if action == 0:
            rob.move_blocking(80, 80, 200)
        elif action == 1:
            rob.move_blocking(-60, 20, 300)
        elif action == 2:
            rob.move_blocking(20, -60, 300)

        irs_new = rob.read_irs()
        next_state_raw = [irs_new[7], irs_new[3], irs_new[4], irs_new[5], irs_new[6]]
        next_state = np.array(next_state_raw) / 200.0

        if isinstance(rob, HardwareRobobo):
            next_state = next_state * 1.5

        curr_wheels = rob.read_wheels()
        left_delta = curr_wheels.wheel_pos_l - prev_wheels.wheel_pos_l
        right_delta = curr_wheels.wheel_pos_r - prev_wheels.wheel_pos_r
        forward_progress = (left_delta + right_delta) / 2.0
        prev_wheels = curr_wheels

        collision = any(next_state_raw[i] > THRESHOLD_COLLISION[i] for i in range(5))
        near_wall = any(next_state_raw[i] > THRESHOLD_NEAR[i] for i in range(5))

        if collision:
            reward = -10.0
            done = False
            print("Collision detected! Episode ends.")
            if isinstance(rob, HardwareRobobo):
                rob.move_blocking(-60, -60, 300)
            
        elif near_wall:
            reward = -1.0
        else:
            reward = forward_progress * 0.01
            if action == 0:
                reward += 5.0
            else:
                reward -= 0.3

        reward -= 0.01

        agent.store_transition(state, action, reward, next_state, done)
        agent.train()
        total_reward += reward

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()
    return total_reward
