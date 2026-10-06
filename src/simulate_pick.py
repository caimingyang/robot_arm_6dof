"""
MuJoCo Simulation: 6-DOF Robot Arm Picking an Apple
====================================================
Visualizes the robot arm and runs a pick sequence with physics.

Usage:
    python simulate_pick.py

Controls:
    - Close viewer window to exit
    - The robot will automatically execute: home -> approach -> grasp -> lift
"""

import time
import math
import numpy as np

import mujoco
import mujoco.viewer

# Pre-computed joint targets (from robot_controller.py IK)
Q_HOME = np.array([0.0, 0.785, -1.571, 0.0, 0.785, 0.0])
Q_APPROACH = np.array([0.401, -0.640, -1.515, 0.576, 0.782, 0.0])
Q_GRASP = np.array([0.401, -0.890, -1.215, 0.576, 0.982, 0.0])
Q_LIFT = np.array([0.401, -0.640, -1.515, 0.576, 0.782, 0.0])

# Gripper positions
GRIPPER_OPEN = 0.035   # 35mm per finger (70mm total)
GRIPPER_CLOSED = 0.005  # 5mm per finger (10mm total, for 75mm apple)


def lerp(a, b, t):
    """Linear interpolation."""
    return a + t * (b - a)


def main():
    print("=" * 50)
    print("MuJoCo Robot Arm - Apple Pick Simulation")
    print("=" * 50)

    # Load model
    model = mujoco.MjModel.from_xml_path("src/robot_scene.xml")
    data = mujoco.MjData(model)

    # Get joint indices
    joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
                   "left_gripper_joint", "right_gripper_joint"]
    joint_ids = []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        joint_ids.append(jid)

    # Get actuator indices
    act_names = ["actuator_j1", "actuator_j2", "actuator_j3", "actuator_j4",
                 "actuator_j5", "actuator_j6",
                 "actuator_left_gripper", "actuator_right_gripper"]
    act_ids = []
    for name in act_names:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        act_ids.append(aid)

    print(f"Joints: {len(joint_ids)}, Actuators: {len(act_ids)}")

    # Simulation state
    phase = 0
    phase_names = ["HOME", "APPROACH", "GRASP", "LIFT", "DONE"]
    phase_durations = [1.0, 2.0, 2.0, 2.0, 1.0]
    phase_start = time.time()

    dt = model.opt.timestep

    # Set initial control
    for i in range(6):
        data.ctrl[act_ids[i]] = Q_HOME[i]
    data.ctrl[act_ids[6]] = GRIPPER_OPEN
    data.ctrl[act_ids[7]] = GRIPPER_OPEN

    print("\nStarting simulation...")
    print("Phases: HOME -> APPROACH -> GRASP -> LIFT -> DONE")
    print("Close viewer window to exit.\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.cam.distance = 1.5
        viewer.cam.lookat[:] = [0.2, 0, 0.3]

        while viewer.is_running():
            step_start = time.time()

            # Phase management
            elapsed = time.time() - phase_start
            duration = phase_durations[phase]
            t = min(elapsed / duration, 1.0) if duration > 0 else 1.0

            # Phase transitions
            if t >= 1.0 and phase < len(phase_durations) - 1:
                phase += 1
                phase_start = time.time()
                t = 0.0
                print(f"-> Phase: {phase_names[phase]}")

            # Interpolate targets based on phase
            if phase == 0:  # HOME
                q_target = Q_HOME
                g_target = GRIPPER_OPEN
            elif phase == 1:  # APPROACH
                q_target = lerp(Q_HOME, Q_APPROACH, t)
                g_target = GRIPPER_OPEN
            elif phase == 2:  # GRASP
                q_target = lerp(Q_APPROACH, Q_GRASP, t)
                g_target = lerp(GRIPPER_OPEN, GRIPPER_CLOSED, t)
            elif phase == 3:  # LIFT
                q_target = lerp(Q_GRASP, Q_LIFT, t)
                g_target = GRIPPER_CLOSED
            else:  # DONE
                q_target = Q_LIFT
                g_target = GRIPPER_CLOSED

            # Apply control
            for i in range(6):
                data.ctrl[act_ids[i]] = q_target[i]
            data.ctrl[act_ids[6]] = g_target
            data.ctrl[act_ids[7]] = g_target

            # Step physics
            mujoco.mj_step(model, data)

            # Sync viewer
            viewer.sync()

            # Real-time pacing
            sleep_time = dt - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    print("\nSimulation complete.")


if __name__ == "__main__":
    main()
