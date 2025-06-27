from lmt1 import run_avoidance_training
from lmt2 import run_foraging_training
from lmt3 import run_secure_training

def run_demo_episode(rob, agent, task):
    """Run a single episode using the trained agent without exploration (epsilon=0.0)."""
    epsilon = 0.01
    if task == 1:
        reward = run_avoidance_training(rob, agent, epsilon)
        print(f"Demo episode finished | Reward: {reward:.2f}")

    if task == 2:
        reward = run_foraging_training(rob, agent, epsilon)
        reward = reward[0]
        print(f"Demo episode finished | Reward: {reward:.2f}")

    if task == 3:
        reward = run_secure_training(rob, agent[0], agent[1], epsilon)
        print(f"Demo episode finished | Reward: {reward:.2f}")