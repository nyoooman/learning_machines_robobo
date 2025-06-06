import cv2
import matplotlib.pyplot as plt
import numpy as np

from data_files import FIGURES_DIR
from robobo_interface import (
    IRobobo,
    Emotion,
    LedId,
    LedColor,
    SoundEmotion,
    SimulationRobobo,
    HardwareRobobo,
)

def plot_sensor_data(sensor_data, title="Sensor Readings Over Time"):
    """
    Plots values from 3 sensory units over a series of steps.

    Parameters:
    - sensor_data (array-like): Shape (n_steps, 3)
    - title (str): Title of the plot
    """
    
    if sensor_data.shape[1] != 3:
        raise ValueError("sensor_data must have 3 columns for 3 sensors.")

    steps = np.arange(sensor_data.shape[0])
    
    plt.figure(figsize=(10, 5))
    plt.plot(steps, sensor_data[:, 0], label="Sensor 1", marker='o')
    plt.plot(steps, sensor_data[:, 1], label="Sensor 2", marker='s')
    plt.plot(steps, sensor_data[:, 2], label="Sensor 3", marker='^')

    plt.xlabel("Step")
    plt.ylabel("Sensor Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("C:\GitHub\learning_machines_robobo\examples\full_project_setup\catkin_ws\src\learning_machines\src\learning_machines\sensor_plot.png")
    plt.close()  # Optional: closes the figure to free memory

# def hardcoded_move(rob: IRobobo):
#     for i in range(5):
#         rob.move_blocking(100, 100, 1000)
#         rob.move_blocking(0, 100, 1000)
#         rob.move_blocking(100, 100, 1000)
#         rob.move_blocking(100, 0, 2000)
#     rob.sleep(1)

def turn_right(rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
        threshold_near = 100

    if isinstance(rob, HardwareRobobo):
        threshold_near = 5   

    max_steps = 50
    step_count = 0
    
    front_l = []
    front_c = []
    front_r = []


    while step_count < max_steps:
        step_count += 1

        irs = rob.read_irs()

        front_left = irs[3]
        print(front_left)
        front_center = irs[4]
        print(front_center)
        front_right = irs[5]
        print(front_right)

        front_l.append(front_left)
        front_c.append(front_center)
        front_r.append(front_right)

        print(f"Step {step_count} | Front-Center IR = {front_center}")

        if front_center > threshold_near:
            print("Obstacle near → turning right...")
            rob.move_blocking(80, -80, 5000)  # Turn right, slower
        else:
            print("Moving forward slowly...")
            rob.move_blocking(80, 80, 300)  # Forward, slower

    print("Task complete.")
    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()

    # prepare data for plot
    steps = np.arange(1,100)
    sensor_data = np.array([front_l, front_c, front_r])

    plot_sensor_data(sensor_data)



def turn_around(rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
        threshold_near = 100

    if isinstance(rob, HardwareRobobo):
        threshold_near = 5   

    max_steps = 50
    step_count = 0
    
    front_l = []
    front_c = []
    front_r = []


    while step_count < max_steps:
        step_count += 1

        irs = rob.read_irs()

        front_left = irs[3]
        print(front_left)
        front_center = irs[4]
        print(front_center)
        front_right = irs[5]
        print(front_right)

        front_l.append(front_left)
        front_c.append(front_center)
        front_r.append(front_right)

        print(f"Step {step_count} | Front-Center IR = {front_center}")

        if front_center > threshold_near:
            print("Wall touched! Going backward...")
            rob.move_blocking(-100, -100, 1000)  # Go backward
            break  # Stop after going back
        else:
            print("Moving forward...")
            rob.move_blocking(100, 100, 300)  # Forward step

    print("Task complete.")
    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()

    # prepare data for plot
    steps = np.arange(1,100)
    sensor_data = np.array([front_l, front_c, front_r])

    plot_sensor_data(sensor_data)