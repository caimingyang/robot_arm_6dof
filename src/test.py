
import argparse
import base64
import io
import json
import math
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

import mujoco
import mujoco.viewer

from config import QWEN_CONFIG

SCENE_XML = "src/robot_scene.xml"
FRAME_SAVE_DIR = "tmp/camera_frames"
CAMERA_RES = (240, 320)



# ============================================================================
# Camera System
# ============================================================================
class CameraSystem:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.renderer = mujoco.Renderer(model, *CAMERA_RES)
        os.makedirs(FRAME_SAVE_DIR, exist_ok=True)
        self.frame_count = 0

    def capture(self, camera_name, save=False):
        self.renderer.update_scene(self.data, camera=camera_name)
        rgb = self.renderer.render()
        if save:
            path = Path(FRAME_SAVE_DIR) / f"{camera_name}_{self.frame_count:04d}.png"
            Image.fromarray(rgb).save(path)
            self.frame_count += 1
        return rgb

    def capture_both(self, save=False):
        eye = self.capture("eye_in_hand", save=save)
        env = self.capture("env_camera", save=save)
        return {"eye_in_hand": eye, "env_camera": env}

    def close(self):
        self.renderer.close()


class RobotEnv:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data = mujoco.MjData(self.model)
        self.cam = CameraSystem(self.model, self.data)

        # Joint qpos addresses
        self.qpos_adr = []
        for i in range(6):
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i+1}")
            self.qpos_adr.append(self.model.jnt_qposadr[jnt_id])
        self.grip_adr = []
        for name in ["left_gripper_joint", "right_gripper_joint"]:
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.grip_adr.append(self.model.jnt_qposadr[jnt_id])

        self.apple_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "apple")
        self.gripper_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper")

        self.joint_angles = np.array([0.0, 0.5, -1.2, 0.0, 0.8, 0.0])
        self.gripper_opening = 0.035
        self.phase = "search"
        self.grasped = False
        self.apple_attached = False
        self._apply_joints()

    def _apply_joints(self):
        for i in range(6):
            self.data.qpos[self.qpos_adr[i]] = self.joint_angles[i]
        self.data.qpos[self.grip_adr[0]] = self.gripper_opening
        self.data.qpos[self.grip_adr[1]] = self.gripper_opening
        if self.apple_attached:
            # Move apple with gripper
            ee = self.data.xpos[self.gripper_body_id].copy()
            self.model.body("apple").pos = ee + np.array([0, 0, -0.06])
        mujoco.mj_forward(self.model, self.data)

    def get_end_effector_pos(self):
        return self.data.xpos[self.gripper_body_id].copy()

    def get_apple_pos(self):
        return self.data.xpos[self.apple_body_id].copy()

    def set_joints(self, q, gripper):
        self.joint_angles = np.copy(q)
        self.gripper_opening = gripper
        self._apply_joints()

    def attach_apple(self):
        if not self.apple_attached:
            self.apple_attached = True
            self._apply_joints()

    def step(self):
        # Apple follows gripper if attached; no physics stepping needed
        if self.apple_attached:
            self._apply_joints()

    def close(self):
        self.cam.close()


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--save-frames", action="store_true")
    args = parser.parse_args()

    env = RobotEnv()


    dt = 0.05
    step_idx = 0

    if args.no_viewer:
        _run(env, args, dt, step_idx, use_viewer=False)
    else:
        _run(env, args, dt, step_idx, use_viewer=True)

    
    env.close()
    print("11111111111111111111111111")
    print("\nDone.")


def _run(env, args, dt, step_idx, use_viewer):
    loop_fn = _loop_with_viewer if use_viewer else _loop_headless
    loop_fn(env, args, dt, step_idx)


def _loop_headless(env, args, dt, step_idx):
    while step_idx < args.steps and not env.grasped:
        t0 = time.time()
        env.step()
        _sleep_remain(t0, dt)
        step_idx += 1
    print(f"\nFinished: {step_idx} steps, phase={env.phase}")


def _loop_with_viewer(env, args, dt, step_idx):
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.cam.distance = 1.5
        viewer.cam.lookat[:] = [0.2, 0, 0.3]

        while viewer.is_running() and step_idx < args.steps and not env.grasped:
            t0 = time.time()
            frames = env.cam.capture_both(save=args.save_frames)
            print('00000000000')
            dq = np.array([0.0, 0.01, 0.01, 0.0, 0.01, 0.0])
            env.joint_angles += dq
            print(env.joint_angles)
            env.set_joints(env.joint_angles, env.gripper_opening)
            env.step()
            viewer.sync()
            _sleep_remain(t0, dt)
            step_idx += 1

    print(f"\nFinished: {step_idx} steps, phase={env.phase}")


def _sleep_remain(t0, target_dt):
    elapsed = time.time() - t0
    sleep = target_dt - elapsed
    if sleep > 0:
        time.sleep(sleep)


if __name__ == "__main__":
    main()
