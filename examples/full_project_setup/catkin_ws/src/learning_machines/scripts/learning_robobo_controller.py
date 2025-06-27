#!/usr/bin/env python3
import sys
import numpy as np
import time

from robobo_interface import IRobobo, SimulationRobobo, HardwareRobobo
from lmt1 import ObstacleAvoidanceAgent, run_avoidance_training
from lmt2 import ForagingAgent, run_foraging_training
from lmt3 import SecuringAgent, run_secure_training
from measure_gap import sensor_gap, camera_gap, determine_sensors
from demo import run_demo_episode

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
        print("Running SimulationRobobo")

    elif "--hardware" in args:
        mode = "hardware"
        rob = HardwareRobobo(camera=True)
        print("Running on Hardware")
    
    else:
        raise ValueError("Specify mode: --hardware/--simulation")

    # Define task: 1/2/3
    if "--task1" in args:
        task = 1   
        agent = ObstacleAvoidanceAgent()
    elif "--task2" in args:
        task = 2
        agent = ForagingAgent()
    elif "--task3" in args:
        task = 3
        agent = SecuringAgent()
        agent2 = SecuringAgent()
    else:
        raise ValueError("Specify task: --task1/--task2/--task3")

    # Consistent model/reward paths
    load_model_path = f"/root/results/task{task}/model_new.pth"
    model_path = f"/root/results/task{task}/model_new2.pth"
    rewards_path = f"/root/results/task{task}/reward_new4.npy"

    # Define mode: Training/Demo/Gap 
    if "--demo" in args:           
        agent.load_model(model_path)
        if task == 3:
            agent2.load_model(load_model_path)
        run_demo_episode(rob, [agent, agent2], task)
        sys.exit(0)

    elif "--training" in args:    
        if isinstance(rob, SimulationRobobo):
            num_episodes = 300
            episode_rewards = []
            if "--load" in args:
                agent.load_model(load_model_path)

            for episode in range(num_episodes):
                epsilon = max(0.05, 0.95 * (0.995 ** episode))
                if task == 1:
                    reward = run_avoidance_training(rob, agent, epsilon)
                    print(f"Episode {episode+1}/{num_episodes} | Epsilon: {epsilon:.2f} | Reward: {reward:.2f}")
                    episode_rewards.append(reward)
                if task == 2:
                    reward = run_foraging_training(rob, agent, epsilon)
                    print(f"Episode {episode+1}/{num_episodes} | Epsilon: {epsilon:.2f} | Reward: {reward[0]:.2f} | Total green: {reward[1]:.2f}")
                    episode_rewards.append(reward[0])
                if task == 3:
                    reward = run_secure_training(rob, agent, agent2, epsilon)
                    print(f"Episode {episode+1}/{num_episodes} | Epsilon: {epsilon:.2f} | Reward: {reward:.2f}")
                    episode_rewards.append(reward)
 
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