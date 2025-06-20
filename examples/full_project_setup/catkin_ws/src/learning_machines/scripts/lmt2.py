import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from robobo_interface import IRobobo, SimulationRobobo, HardwareRobobo

# === Hyperparameters ===
LEARNING_RATE = 0.001
GAMMA = 0.99
BATCH_SIZE = 64
REPLAY_MEMORY_SIZE = 10000
TARGET_UPDATE_FREQ = 100
NUM_ACTIONS = 3
OBSERVATION_SPACE = 1

# === Reward weights ===
REWARDS = {
    "green_seen": 30.0,
    "green_closer": 2.0,
    "food_collected": 50.0,
    "forward_progress_factor": 0.05,
    "spin_penalty": -1.0,
    "no_green_penalty": -8.0,
    "stuck_penalty": -2.0,
    "step_penalty": -0.01,
}

# === DQN Model ===
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(OBSERVATION_SPACE, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, NUM_ACTIONS)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# === Agent ===
class ForagingAgent:
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
        with torch.no_grad():
            return self.policy_net(torch.FloatTensor(state).unsqueeze(0)).argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        q_values = self.policy_net(states).gather(1, actions).squeeze()
        next_q = self.target_net(next_states).max(1)[0].detach()
        expected_q = rewards + GAMMA * next_q * (1 - dones)
        loss = nn.MSELoss()(q_values, expected_q)
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

# === Green Detection ===
def detect_green_percentage(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Simulation RGB range
        lower_green = np.array([0, 50, 50])
        upper_green = np.array([80, 255, 255])

    # # Hardware RGB range
    #     lower_green = np.array([20, 70, 70])
    #     upper_green = np.array([100, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    return np.sum(mask) / (mask.size * 255.0)

# === Main Training Loop ===
def run_foraging_training(rob: IRobobo, agent: ForagingAgent, epsilon: float):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    rob.set_phone_tilt(95, 50)
    
    total_reward = 0.0
    step_count = 0
    max_steps = 600
    prev_wheels = rob.read_wheels()
    food_collected = rob.get_nr_food_collected()
    green_history = deque(maxlen=10)
    stuck_counter = 0
    green_signal_sum = 0.0

    while step_count < max_steps:
        step_count += 1

        # Read state
        image = rob.read_image_front()
        green_signal = detect_green_percentage(image)

        # if isinstance(rob, HardwareRobobo):
        #     green_signal *= 1

        green_signal_sum += green_signal
        green_history.append(green_signal)
        state = np.array([green_signal])
        action = agent.select_action(state, epsilon)

        # Take action
        if action == 0:
            rob.move_blocking(80, 80, 300)
        elif action == 1:
            rob.move_blocking(-60, 60, 250)
        elif action == 2:
            rob.move_blocking(60, -60, 250)

        # Read next state
        image = rob.read_image_front()
        green_signal_next = detect_green_percentage(image)

        # if isinstance(rob, HardwareRobobo):
        #     green_signal_next *= 1

        next_state = np.array([green_signal_next])

        # Movement metrics
        curr_wheels = rob.read_wheels()
        left_delta = curr_wheels.wheel_pos_l - prev_wheels.wheel_pos_l
        right_delta = curr_wheels.wheel_pos_r - prev_wheels.wheel_pos_r
        forward_progress = (left_delta + right_delta) / 2.0
        spin_amount = abs(left_delta - right_delta)
        prev_wheels = curr_wheels

        # === Reward calculation ===
        reward = 0.0

        # Reward/penalty for detecting (no) green
        if green_signal > 0.002:
            reward += REWARDS["green_seen"] * green_signal
        else:
            reward += REWARDS["no_green_penalty"]

        # Dynamic reward for increased green compared to previous step
        if len(green_history) >= 1 and green_signal > green_history[-1] and action == 0:
            reward += REWARDS["green_closer"]

        # Reward that encourages forward movement
        reward += forward_progress * REWARDS["forward_progress_factor"]

        # Penalty that penalizes excessive spinning
        if forward_progress < 0.2 and spin_amount > 5.0:
            reward += REWARDS["spin_penalty"]

        # Reward for collecting food
        new_collected = rob.get_nr_food_collected()
        if new_collected > food_collected:
            reward += REWARDS["food_collected"]
            food_collected = new_collected

        # Penalty for getting stuck
        if forward_progress < 0.2:
            stuck_counter += 1
        else:
            stuck_counter = 0
        if stuck_counter >= 15:
            reward += REWARDS["stuck_penalty"]
            stuck_counter = 0

        # Small penalty for every step taken
        reward += REWARDS["step_penalty"]

        agent.store_transition(state, action, reward, next_state, False)
        agent.train()
        total_reward += reward

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()

    return total_reward, green_signal_sum