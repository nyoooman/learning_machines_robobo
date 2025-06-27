import cv2
import time

from robobo_interface import IRobobo, SimulationRobobo, HardwareRobobo
from lmt2 import detect_green_percentage
from lmt3 import detect_colored_regions

def sensor_gap(rob: IRobobo):
    # determine robobo instance and start simulation if necessary
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
        instance = "simulation"
        iterations = 1
    else:
        instance = "reality"
        iterations = 10

    # define sensors and dictionary with sensor data
    sensors = ["LL", "L", "C", "R", "RR"]
    sensor_data = {}

    for iteration in range(iterations):    
        if isinstance(rob, HardwareRobobo):
            print(f"Iteration{iteration + 1}/{iterations}: Resetting robot...")
            time.sleep(5)
            sensor_data[f"C_{iteration}"] = []

        # measure sensor data over iterations until robobo hits wall
        for i in range(22):
            if instance == "simulation":
                rob.move_blocking(50, 50, 100)
                print(f"Rob moving, step {i}")

            if instance == "reality":
                rob.move_blocking(100, 100, 200)
                print(f"Rob moving, step {i}")

            irs = rob.read_irs()
            sensor_data[f"C_{iteration}"].append(round(irs[4], 2))     

    # save dictionary to text file 
    with open(f"/root/results/sensor_values/{instance}.txt", "w") as output:
        for key, value in sensor_data.items():
            output.write(f'{key}: {value}\n')

    # stop simulation if necessary
    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()
    return

def camera_gap(rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    # set phone to tilt at different angles
    rob.set_phone_tilt_blocking(109, 100)

    for i in range(10):
        # give camera time to start
        rob.move_blocking(-80,40,200)

        # take image and save image
        image = rob.read_image_front()

        # determine percentage
        green_zones = detect_colored_regions(image, "g")
        red_zones = detect_colored_regions(image,"r")
        print(f"Green: {green_zones}, Red: {red_zones}")

        cv2.imwrite(f"/root/results/images/SimulationPOV{i}.jpg", image)
        print("Image saved successfully.")


    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()
    return

def determine_sensors(rob:IRobobo):
    for i in range(100):
        time.sleep(1)
        irs_new = rob.read_irs()
        print(f"Sensor 0: {irs_new[0]}") 
        "LB"
        print(f"Sensor 1: {irs_new[1]}")
        "RB"
        print(f"Sensor 2: {irs_new[2]}")
        "L"
        print(f"Sensor 3: {irs_new[3]}")
        "R"
        print(f"Sensor 4: {irs_new[4]}")
        "C"
        print(f"Sensor 5: {irs_new[5]}")
        "RR"
        print(f"Sensor 6: {irs_new[6]}")
        "CB"
        print(f"Sensor 7: {irs_new[7]}")
        "L"
    return   