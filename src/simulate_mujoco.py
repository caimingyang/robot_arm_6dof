"""
MuJoCo Simulation for 6-DOF Robot Arm
=====================================
Loads the URDF, adds an apple, and simulates the pick motion.

Usage:
    python simulate_mujoco.py
"""

import time
import math
import numpy as np

# Try MuJoCo
try:
    import mujoco
    import mujoco.viewer
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    print("MuJoCo not available. Please install: uv pip install mujoco")
    exit(1)

# Apple target position (meters)
APPLE_POS = [0.35, 0.0, 0.08]


def create_mjcf_with_apple(urdf_path):
    """
    Wrap the URDF in an MJCF that includes ground plane, lighting, and an apple.
    MuJoCo can compile URDF directly via its compiler.
    """
    mjcf = f"""<mujoco model="robot_arm_scene">
  <compiler meshdir="." texturedir="." autolimits="true"/>

  <option timestep="0.002" gravity="0 0 -9.81"/>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.6 0.8 1.0" rgb2="0.2 0.4 0.8"
             width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.9 0.9 0.9" rgb2="0.7 0.7 0.7"
             width="512" height="512"/>
    <material name="grid_mat" texture="grid" texrepeat="10 10" reflectance="0.1"/>
    <material name="apple_red" rgba="0.9 0.2 0.1 1.0"/>
    <material name="apple_green" rgba="0.3 0.7 0.2 1.0"/>
  </asset>

  <worldbody>
    <!-- Ground plane -->
    <geom name="floor" type="plane" size="2 2 0.1" material="grid_mat"/>

    <!-- Lighting -->
    <light name="sun" pos="1 1 3" dir="-1 -1 -3" diffuse="0.8 0.8 0.8"/>
    <light name="fill" pos="-1 0.5 2" dir="1 -0.5 -2" diffuse="0.4 0.4 0.4"/>

    <!-- Apple -->
    <body name="apple" pos="{APPLE_POS[0]} {APPLE_POS[1]} {APPLE_POS[2]}">
      <freejoint/>
      <geom name="apple_geom" type="sphere" size="0.0375" mass="0.15"
            material="apple_red" friction="0.8 0.02 0.01"/>
      <!-- Stem -->
      <geom name="apple_stem" type="cylinder" size="0.003 0.015"
            pos="0 0 0.045" material="apple_green"/>
    </body>

    <!-- Table (under apple) -->
    <body name="table" pos="0.35 0 0.04">
      <geom name="table_top" type="box" size="0.15 0.15 0.04"
            rgba="0.5 0.35 0.2 1.0" friction="0.8"/>
    </body>

    <!-- Include the robot URDF -->
    <include file="{urdf_path}"/>
  </worldbody>
</mujoco>"""
    return mjcf


def compute_joint_targets_ik(target_pos, current_q, dh_params, limits):
    """
    Simple numerical IK (same logic as robot_controller.py)
    """
    q = np.copy(current_q)
    damping = 0.1
    max_iter = 100
    tol = 1e-4

    for _ in range(max_iter):
        # Forward kinematics to get current position
        T = np.eye(4)
        for i, (a, alpha, d, theta_offset) in enumerate(dh_params):
            theta = theta_offset + q[i]
            ct, st = math.cos(theta), math.sin(theta)
            ca, sa = math.cos(alpha), math.sin(alpha)
            Ti = np.array([
                [ct, -st*ca,  st*sa, a*ct],
                [st,  ct*ca, -ct*sa, a*st],
                [0,   sa,     ca,    d    ],
                [0,   0,      0,     1    ]
            ])
            T = T @ Ti

        pos = T[:3, 3]
        err = target_pos - pos

        if np.linalg.norm(err) < tol:
            break

        # Numerical Jacobian (position only, 3x6)
        J = np.zeros((3, 6))
        for i in range(6):
            dq = np.copy(q)
            dq[i] += 1e-6
            T2 = np.eye(4)
            for j, (a, alpha, d, theta_offset) in enumerate(dh_params):
                theta = theta_offset + dq[j]
                ct, st = math.cos(theta), math.sin(theta)
                ca, sa = math.cos(alpha), math.sin(alpha)
                Ti = np.array([
                    [ct, -st*ca,  st*sa, a*ct],
                    [st,  ct*ca, -ct*sa, a*st],
                    [0,   sa,     ca,    d    ],
                    [0,   0,      0,     1    ]
                ])
                T2 = T2 @ Ti
            pos2 = T2[:3, 3]
            J[:, i] = (pos2 - pos) / 1e-6

        JtJ = J.T @ J
        damped = JtJ + damping**2 * np.eye(6)
        dq = np.linalg.solve(damped, J.T @ err)
        q += dq * 0.5

        for i in range(6):
            lo, hi = limits[i]
            q[i] = np.clip(q[i], lo, hi)

    return q


def main():
    print("=" * 50)
    print("MuJoCo Robot Arm Simulation")
    print("=" * 50)

    urdf_path = "src/robot_arm_6dof.urdf"

    # MuJoCo can't directly load URDF via from_xml_path in the same way.
    # Instead, we use the compiler to convert URDF to MJCF.
    # Alternative: use mujoco.MjModel.from_xml_string with include.

    # For simplicity, let's create a combined MJCF that references the URDF.
    # Note: MuJoCo's include works with MJCF, not URDF.
    # We need to convert URDF -> MJCF first, or load URDF differently.

    # MuJoCo 3.x approach: use mujoco.MjModel.from_xml_path with .urdf extension
    # The compiler will auto-convert URDF to internal MJCF.
    try:
        model = mujoco.MjModel.from_xml_path(urdf_path)
        print(f"Loaded URDF: {urdf_path}")
    except Exception as e:
        print(f"Failed to load URDF directly: {e}")
        print("\nTrying MJCF wrapper approach...")

        # Create an MJCF that includes the URDF via <body> with <include>
        # Actually MuJoCo doesn't support including URDF in MJCF.
        # We need to manually read URDF and embed it, or use a different approach.

        # Let's read the URDF and try to parse it
        import xml.etree.ElementTree as ET
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        # Build a minimal MJCF from URDF elements
        # This is a simplified conversion
        links = {}
        joints = []
        for link in root.findall('link'):
            name = link.get('name')
            links[name] = link
        for joint in root.findall('joint'):
            joints.append(joint)

        print(f"Found {len(links)} links, {len(joints)} joints")
        print("\nNote: Full MuJoCo URDF import requires manual conversion or ROS tools.")
        print("Recommended: Use Gazebo or PyBullet for direct URDF simulation.")
        return

    data = mujoco.MjData(model)

    # Get joint info
    n_joints = model.njnt
    print(f"Model joints: {n_joints}")
    for i in range(n_joints):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        print(f"  [{i}] {name}")

    # Simulation parameters
    dt = model.opt.timestep
    duration = 5.0  # seconds
    steps = int(duration / dt)

    # Joint angle targets for apple approach
    # Pre-computed from robot_controller.py
    q_approach = np.array([0.401, -0.640, -1.515, 0.576, 0.782, 0.0])
    q_grasp = np.array([0.401, -0.890, -1.215, 0.576, 0.982, 0.0])

    print(f"\nRunning simulation for {duration}s...")
    print("Close viewer window to exit.")

    # Run simulation with viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        phase = 0  # 0=home, 1=approach, 2=grasp, 3=hold
        phase_start = start_time

        while viewer.is_running() and time.time() - start_time < duration:
            step_start = time.time()

            # Simple joint control via position servos
            elapsed = time.time() - phase_start

            if phase == 0 and elapsed > 1.0:
                phase = 1
                phase_start = time.time()
                print("-> Approach")
            elif phase == 1 and elapsed > 1.5:
                phase = 2
                phase_start = time.time()
                print("-> Grasp")
            elif phase == 2 and elapsed > 1.5:
                phase = 3
                phase_start = time.time()
                print("-> Hold")

            # Set joint targets
            if phase == 0:
                target = np.zeros(6)
            elif phase == 1:
                target = q_approach
            elif phase == 2:
                target = q_grasp
            else:
                target = q_grasp

            # Apply to actuators (if any) or directly set ctrl
            for i in range(min(6, model.nu)):
                data.ctrl[i] = target[i]

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
