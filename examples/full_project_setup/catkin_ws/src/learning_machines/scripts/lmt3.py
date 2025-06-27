import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import time

from robobo_interface import IRobobo, SimulationRobobo, HardwareRobobo

# === Hyperparameters ===
LEARNING_RATE = 0.001
GAMMA = 0.99
BATCH_SIZE = 64
REPLAY_MEMORY_SIZE = 10000
TARGET_UPDATE_FREQ = 100
NUM_ACTIONS = 5
OBSERVATION_SPACE = 8

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
class SecuringAgent:
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

# === Secure_Phase ===
RED_REWARDS = {
    "no_red": -8.0,
    "red_multiplier": np.array([1, 1, 2, 4]),
    "red_seen": 2.0,
    # "approach": 3.0,
    "securing": 5.0,
    "move_forward": 1.0, 
    # "orientate": 1.0,
    "secured": 50.0
}

# === Deliver_Phase ===
GREEN_REWARDS = {
    "delivery_phase": 3.0,
    "no_green": -4.0,
    "green_multiplier": np.array([2, 2, 4, 8]),
    "green_seen": 2.0,
    # "approach": 3.0,
    "move_forward": 1.0,
    # "orientate": 1.0,
    "lost_package": -40.0,
    "delivered": 150.0
}

def detect_colored_regions(image, color, mode):
    """
    Detect red pixels counts in 5 regions. Returns array as (L, R, CTop, CMiddle, CBottom)
    """

    if image is None:
        raise ValueError("Image is None")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Red color ranges in HSV
    if color == "r":
        lower1 = np.array([0, 100, 100])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([160, 100, 100])
        upper2 = np.array([179, 255, 255])

        # Create colored mask
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
    
    else:
        lower = np.array([45, 100, 50])
        upper = np.array([75, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)


    height, width = mask.shape
    third_w = width // 3
    third_h = height // 3

    # Region masks
    left_mask = mask[:, :third_w]
    right_mask = mask[:, 2*third_w:]

    center_x_start = third_w
    center_x_end = 2 * third_w

    if mode == "h":
        center_top = mask[0:third_h, center_x_start:center_x_end]
        center_bot = mask[third_h:, center_x_start:center_x_end]

    else:
        center_top = mask[0:2*third_h, center_x_start:center_x_end]
        center_bot = mask[2*third_h:, center_x_start:center_x_end]


    # Count red pixels (nonzero mask values)
    counts = np.array([
        np.count_nonzero(left_mask),
        np.count_nonzero(right_mask),
        np.count_nonzero(center_top),
        np.count_nonzero(center_bot)
    ])

    clipped_counts = np.minimum(counts, 10000)

    return clipped_counts

def run_secure_training(rob: IRobobo, agent, agent_green, epsilon):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    rob.set_phone_tilt_blocking(109, 100)

    secured = False
    delivered = False
    total_reward = 0.0
    step_count = 0
    max_steps = 500
    docking = 0

    while step_count < max_steps:
        step_count += 1

        # Read state
        image = cv2.rotate(rob.read_image_front(), cv2.ROTATE_180)

        if isinstance(rob, SimulationRobobo):
            green_mask = detect_colored_regions(image, "g", "s") / 10000
            red_mask = detect_colored_regions(image, "r", "s") / 10000

        else:
            green_mask = detect_colored_regions(image, "g", "h") / 10000
            red_mask = detect_colored_regions(image, "r", "h") / 10000

        state = np.hstack((green_mask, red_mask))

        if secured:
            epsilon += 0.01
            epsilon = min(0.95, epsilon)
            action = agent_green.select_action(state, epsilon)

        else:
            action = agent.select_action(state, epsilon)

        # Take action
        if action == 0:
            rob.move_blocking(80, 80, 300)
        elif action == 1:
            rob.move_blocking(20, 50, 300)
        elif action == 2:
            rob.move_blocking(50, 20, 300)
        elif action == 3:
            rob.move_blocking(-60, 60, 300)
        elif action == 4:
            rob.move_blocking(60, -60, 300)

        # Read next state
        image = cv2.rotate(rob.read_image_front(), cv2.ROTATE_180)

        if isinstance(rob, SimulationRobobo):
            green_mask_next = detect_colored_regions(image, "g", "s") / 10000
            red_mask_next = detect_colored_regions(image, "r", "s") / 10000

        else:
            green_mask_next = detect_colored_regions(image, "g", "h") / 10000
            red_mask_next = detect_colored_regions(image, "r", "h") / 10000

        next_state = np.hstack((green_mask_next, red_mask_next))

        # === Reward calculation ===
        reward = 0.0

        # === Phase 1 | Securing ===
        if not secured:
            if sum(red_mask_next) == 0:
                reward += RED_REWARDS["no_red"]
                # if action >= 3:
                #     reward += RED_REWARDS["orientate"]
            else:
                reward += RED_REWARDS["red_seen"]
                reward += sum(red_mask_next * RED_REWARDS["red_multiplier"])
                # if action < 3:
                #     reward += RED_REWARDS["approach"]

            if red_mask_next[3] == 1:
                docking += 1
                if action == 0:
                    reward += RED_REWARDS["securing"]
                    docking += 1
            else:
                docking = 0

            if action == 0:
                reward += RED_REWARDS["move_forward"]

            if docking >= 3:
                # print("package_secured")
                docking = 0
                secured = True
                reward += RED_REWARDS["secured"]
        
        # === Phase 2 | Delivering ===
        else:
            if red_mask_next[3] * 1.1 < sum(red_mask_next[0:2]):
                secured = False
                # print("Lost package")
                reward += GREEN_REWARDS["lost_package"]

            else:
                reward += GREEN_REWARDS["delivery_phase"]

        if secured:
            if sum(green_mask_next) == 0:
                reward += GREEN_REWARDS["no_green"]
                # if action >= 3:
                #     reward += GREEN_REWARDS["orientate"]
            else:
                reward += GREEN_REWARDS["green_seen"]
                reward += sum(green_mask_next * GREEN_REWARDS["green_multiplier"])
                # if action < 3:
                #     reward += GREEN_REWARDS["approach"]

            if action == 0:
                reward += GREEN_REWARDS["move_forward"]

            if isinstance(rob, SimulationRobobo):
                if rob.base_detects_food():
                    reward += GREEN_REWARDS["delivered"]
                    delivered = True

            if delivered:
                reward += (max_steps - step_count) * 5

        red_ratio = [sum(red_mask_next[0:2]), red_mask_next[3]]
        green_ratio = [sum(green_mask_next[0:2]), green_mask_next[3]]

        red_rounded = [round(x, 1) for x in red_ratio]
        green_rounded = [round(x, 1) for x in green_ratio]

        print(f"Red: {red_rounded}, Green: {green_rounded}, Reward: {reward}, Secured: {secured}")

        agent.store_transition(state, action, reward, next_state, False)
        agent.train()
        total_reward += reward

        if delivered:
            print("package_delivered!")
            break

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()

    return total_reward
