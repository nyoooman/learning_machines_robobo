#!/usr/bin/env python3
import sys
import numpy as np
import time

from robobo_interface import IRobobo, SimulationRobobo, HardwareRobobo
from lmt1 import ObstacleAvoidanceAgent, run_avoidance_training
from lmt2 import ForagingAgent, run_foraging_training
from measure_gap import sensor_gap, camera_gap, determine_sensors

def run_demo_episode(rob, agent, task):
    """Run a single episode using the trained agent without exploration (epsilon=0.0)."""
    epsilon = 0.0
    if task == "task1":
        reward = run_avoidance_training(rob, agent, epsilon)
        print(f"Demo episode finished | Reward: {reward:.2f}")

    if task == "task2":
        reward = run_foraging_training(rob, agent, epsilon)
        reward = reward[0]
        print(f"Demo episode finished | Reward: {reward:.2f}")
    

if __name__ == "__main__":
    args = sys.argv[1:]
    print(args)
    if not args:
        raise ValueError(
            """Usage:
            --simulation [id]
            --hardware'
            --task1
            --task2
            --gap
            --training
            --demo --sim [id]
            --demo --hardware"""
        )

    mode = None

    # === Parse arguments ===  
    # Define Robobo: Simulation/Hardware 
    if "--simulation" in args:
        mode = "simulation"
        rob = SimulationRobobo()
        print(f"Running SimulationRobobo")

    elif "--hardware" in args:
        mode = "hardware"
        rob = HardwareRobobo(camera=True)
        print("Running on Hardware")
    
    else:
        raise ValueError("Specify mode: --hardware/--simulation")

    # Define task: 1/2/3
    if "--task1" in args:
        task = "task1"
        # task 1 model and reward paths
        model_path = f"/root/results/task1_model.pth"
        rewards_path = f"/root/results/task1_reward.npy"
        agent = ObstacleAvoidanceAgent()
    
    elif "--task2" in args:
        task = "task2"
        # # Consistent model/reward paths
        load_model_path = f"/root/results/task2_model.pth"
        model_path = f"/root/results/task2_model_new.pth"
        rewards_path = f"/root/results/task2_reward.npy"
        agent = ForagingAgent()
    else:
        raise ValueError("Specify task: --task1/--task2")

    # Define mode: Training/Demo/Gap 
    if "--demo" in args:           
        agent.load_model(model_path)
        run_demo_episode(rob, agent, task)
        sys.exit(0)

    elif "--training" in args:    
        if isinstance(rob, SimulationRobobo):
            num_episodes = 50
            episode_rewards = []
            if "--load" in args:
                agent.load_model(load_model_path)

            for episode in range(num_episodes):
                epsilon = max(0.01, 0.3 * (0.98 ** episode))
                if task == "task1":
                    reward = run_avoidance_training(rob, agent, epsilon)
                    print(f"Episode {episode+1}/{num_episodes} | Epsilon: {epsilon:.2f} | Reward: {reward:.2f}")
                    episode_rewards.append(reward)
                if task == "task2":
                    reward = run_foraging_training(rob, agent, epsilon)
                    print(f"Episode {episode+1}/{num_episodes} | Epsilon: {epsilon:.2f} | Reward: {reward[0]:.2f} | Total green: {reward[1]:.2f}")
                    episode_rewards.append(reward[0])
 
        # Optional summary
        avg_reward = sum(episode_rewards) / num_episodes
        print(f"Training complete. Average reward over {num_episodes} episodes: {avg_reward:.2f}")
        # Save model and rewards once, after training
        agent.save_model(model_path)
        print(f"Training complete. Model saved to {model_path}")

        np.save(rewards_path, episode_rewards)
        print(f"Rewards saved to {rewards_path}")

    # === Reality Gap ===
    elif "--gap" in args and "--task1" in args:
        determine_sensors(rob)

    elif '--gap' in args and "--task2" in args:
        camera_gap(rob)

    else:
        raise ValueError("Specify mode: --demo/--training/--gap")