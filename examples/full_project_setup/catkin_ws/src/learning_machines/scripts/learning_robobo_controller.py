#!/usr/bin/env python3
import sys
import numpy as np
import time

from robobo_interface import SimulationRobobo, HardwareRobobo
from lmt1 import ObstacleAvoidanceAgent, run_training_episode

def run_demo_episode(rob, agent):
    """Run a single episode using the trained agent without exploration (epsilon=0.0)."""
    epsilon = 0.0
    reward = run_training_episode(rob, agent, epsilon)
    print(f"Demo episode finished | Reward: {reward:.2f}")

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        raise ValueError(
            """Usage:
            --simulation [id]
            --hardware
            --demo --sim [id]
            --demo --hardware"""
        )

    mode = None
    is_demo = False
    identifier = 0

    # Parse arguments
    if "--demo" in args:
        is_demo = True
        mode = "hardware" if "--hardware" in args else "simulation"
        if mode == "simulation":
            try:
                # identifier = int(args[args.index("--sim") + 1])
                identifier = 1
            except (ValueError, IndexError):
                identifier = 0
    elif "--simulation" in args:
        mode = "simulation"
        try:
            identifier = int(args[args.index("--simulation") + 1])
        except (ValueError, IndexError):
            identifier = 0
    elif "--hardware" in args:
        mode = "hardware"
    else:
        raise ValueError("Invalid arguments.")

    # Consistent model/reward paths
    model_path = f"/root/results/task1_dqn_robobo2.pth"
    rewards_path = f"/root/results/rewards_task1_robobo{identifier}.npy"

    # Initialize robot
    if mode == "simulation":
        rob = SimulationRobobo(identifier=identifier)
        print(f"Running SimulationRobobo with identifier {identifier}")
    elif mode == "hardware":
        rob = HardwareRobobo(camera=True)
        print("Running on Hardware")

    # demo
    if is_demo:
        agent = ObstacleAvoidanceAgent()
        agent.load_model(model_path)
        run_demo_episode(rob, agent)
        sys.exit(0)

    # training 
    if isinstance(rob, SimulationRobobo):
        agent = ObstacleAvoidanceAgent()
        num_episodes = 200
        episode_rewards = []

        for episode in range(num_episodes):
            epsilon = max(0.01, 0.95 * (0.995 ** episode))
            reward = run_training_episode(rob, agent, epsilon)
            print(f"Episode {episode+1}/{num_episodes} | Epsilon: {epsilon:.2f} | Reward: {reward:.2f}")
            episode_rewards.append(reward)

    # hybrid TODO
    if isinstance(rob, HardwareRobobo):
        agent = ObstacleAvoidanceAgent()
        agent.load_model(model_path)
        num_episodes = 50
        episode_rewards = []

        for episode in range(num_episodes):
            epsilon = max(0.01, 0.25 * (1 - (episode / num_episodes)))
            print(f"Episode {episode + 1}/{num_episodes}: Resetting robot...")
            time.sleep(5)

            # Run training
            reward = run_training_episode(rob, agent, epsilon)
            print(f"Episode {episode+1}/{num_episodes} | Epsilon: {epsilon:.2f} | Reward: {reward:.2f}")
            episode_rewards.append(reward)

    avg_reward = sum(episode_rewards) / num_episodes
    print(f"Training complete. Average reward over {num_episodes} episodes: {avg_reward:.2f}")
    # Save model
    agent.save_model(model_path)
    print(f"Training complete. Model saved to {model_path}")

    np.save(rewards_path, episode_rewards)
    print(f"Rewards saved to {rewards_path}")
