"""
VLM-Based Robot Arm Controller (Vision-Language-Action)
========================================================
Closed-loop system: camera perceives -> VLM decides -> robot executes.

Uses kinematic control (direct joint positioning) for stable,
predictable arm motion. Apple physics handled separately.

Usage:
    # Mock VLM mode (no API key)
    .venv/Scripts/python.exe src/vlm_robot_controller.py --no-viewer

    # With OpenAI GPT-4V (set OPENAI_API_KEY first)
    .venv/Scripts/python.exe src/vlm_robot_controller.py --vlm openai

    # With Alibaba Cloud Qwen (configured in src/config.py)
    .venv/Scripts/python.exe src/vlm_robot_controller.py --vlm qwen
"""

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

JOINT_LIMITS = [
    (-math.pi, math.pi),
    (-math.pi / 2, math.pi / 2),
    (-2.0, 2.0),
    (-math.pi, math.pi),
    (-math.pi / 2, math.pi / 2),
    (-math.pi, math.pi),
]


def lerp(a, b, t):
    return a + t * (b - a)


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


# ============================================================================
# Vision Processor (Local Stereo Perception)
# ============================================================================
class VisionProcessor:
    """Detect red apple in camera images and estimate its 3D position.
    Uses pixel-to-ray projection + plane intersection for depth estimation.
    """

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.res_h, self.res_w = CAMERA_RES

    def detect_red_object(self, rgb, score_thresh=80, grow_ratio=0.7, min_pixels=10, max_pixels=5000):
        """Detect red apple using score-based flood fill."""
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        red_score = r.astype(float) - 0.5 * g.astype(float) - 0.5 * b.astype(float)
        max_idx = np.unravel_index(np.argmax(red_score), red_score.shape)
        cy, cx = max_idx
        max_score = red_score[cy, cx]
        if max_score < score_thresh:
            return None

        for ratio in [grow_ratio, grow_ratio + 0.1, grow_ratio + 0.2, grow_ratio + 0.3]:
            threshold = max(max_score * ratio, score_thresh * 0.8)
            mask = red_score > threshold
            h, w = mask.shape
            visited = np.zeros_like(mask, dtype=bool)
            queue = [(cx, cy)]
            visited[cy, cx] = True
            component = [(cx, cy)]
            while queue:
                x, y = queue.pop(0)
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        component.append((nx, ny))
                        queue.append((nx, ny))
            if min_pixels <= len(component) <= max_pixels:
                cx_m = int(np.mean([p[0] for p in component]))
                cy_m = int(np.mean([p[1] for p in component]))
                return {"center": (cx_m, cy_m), "pixels": len(component), "max_score": max_score}
        return None

    def pixel_to_world_ray(self, camera_name, cx, cy):
        """Convert pixel coordinate to world-space ray."""
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        fovy = self.model.cam_fovy[cam_id] * np.pi / 180
        x_ndc = (cx - self.res_w / 2) / (self.res_w / 2)
        y_ndc = -(cy - self.res_h / 2) / (self.res_h / 2)
        tan_fovy2 = np.tan(fovy / 2)
        aspect = self.res_w / self.res_h
        dx = x_ndc * tan_fovy2 * aspect
        dy = y_ndc * tan_fovy2
        dz = -1.0
        ray_cam = np.array([dx, dy, dz])
        ray_cam = ray_cam / np.linalg.norm(ray_cam)
        R = self.data.cam_xmat[cam_id].reshape(3, 3)
        ray_world = R @ ray_cam
        cam_pos = self.data.cam_xpos[cam_id].copy()
        return cam_pos, ray_world

    def estimate_apple_position(self, rgb_dict):
        """Estimate apple 3D position from both cameras."""
        estimates = []
        weights = []
        env_det = None
        eye_det = None

        # Primary: env_camera (always sees the apple reliably)
        env_rgb = rgb_dict.get("env_camera")
        if env_rgb is not None:
            env_det = self.detect_red_object(env_rgb, score_thresh=80)
            if env_det and env_det["max_score"] > 100:
                origin, direction = self.pixel_to_world_ray("env_camera", *env_det["center"])
                if abs(direction[2]) > 1e-6:
                    t = (0.0775 - origin[2]) / direction[2]
                    if t > 0:
                        est = origin + t * direction
                        estimates.append(est)
                        weights.append(2.0)

        # Secondary: eye_in_hand (use only when confident)
        eye_rgb = rgb_dict.get("eye_in_hand")
        if eye_rgb is not None:
            eye_det = self.detect_red_object(eye_rgb, score_thresh=100)
            if eye_det and eye_det["max_score"] > 120:
                origin, direction = self.pixel_to_world_ray("eye_in_hand", *eye_det["center"])
                if abs(direction[2]) > 1e-6:
                    t = (0.0775 - origin[2]) / direction[2]
                    if t > 0:
                        est = origin + t * direction
                        estimates.append(est)
                        weights.append(1.0)

        if estimates:
            total_weight = sum(weights)
            avg = sum(w * e for w, e in zip(weights, estimates)) / total_weight
            return avg, env_det, eye_det
        return None, env_det, eye_det

    def format_observation(self, rgb_dict, ee_pos):
        """Format visual detection results as text for the LLM."""
        est, env_det, eye_det = self.estimate_apple_position(rgb_dict)
        lines = []
        lines.append("Visual perception (from camera images):")

        if env_det:
            cx, cy = env_det["center"]
            lines.append(f"  env_camera sees red apple at pixel ({cx}, {cy}), size={env_det['pixels']} px")
        else:
            lines.append("  env_camera: red apple not visible")

        if eye_det:
            cx, cy = eye_det["center"]
            lines.append(f"  eye_in_hand sees red apple at pixel ({cx}, {cy}), size={eye_det['pixels']} px")
        else:
            lines.append("  eye_in_hand: red apple not visible")

        if est is not None:
            rel = est - np.array(ee_pos)
            dist = np.linalg.norm(rel)
            lines.append(f"  Estimated apple position: [{est[0]:.3f}, {est[1]:.3f}, {est[2]:.3f}] m")
            lines.append(f"  Relative to gripper:      [{rel[0]:.3f}, {rel[1]:.3f}, {rel[2]:.3f}] m")
            lines.append(f"  Visual estimated distance: {dist:.3f} m")
        else:
            lines.append("  Unable to estimate apple position from cameras.")

        return "\n".join(lines)


# ============================================================================
# VLM Interfaces
# ============================================================================
class MockVLM:
    """Rule-based VLM: follows a pre-planned joint trajectory to the apple."""

    # Pre-planned waypoints: [j1, j2, j3, j4, j5, j6]
    # Generated via numerical IK in MuJoCo to reach apple at [0.35, 0, 0.0775]
    WAYPOINTS = [
        [0.0, 0.5, -1.2, 0.0, 0.8, 0.0],       # home
        [0.0, 0.5292, -0.6397, 0.0, 0.9542, 0.0],  # approach 1
        [0.0, 0.5584, -0.0794, 0.0, 1.1083, 0.0],  # approach 2
        [0.0, 0.5876, 0.481, 0.0, 1.2625, 0.0],    # approach 3
        [0.0, 0.6168, 1.0413, 0.0, 1.4166, 0.0],   # near apple
        [0.0, 0.6459, 1.6016, 0.0, 1.5708, 0.0],   # grasp position
    ]

    def __init__(self):
        self.waypoint_idx = 0

    def decide(self, rgb_dict, state):
        if state.get("phase") == "done":
            return {"action": "hold", "reasoning": "Task complete."}

        if self.waypoint_idx >= len(self.WAYPOINTS):
            return {"action": "grasp", "target_q": self.WAYPOINTS[-1], "gripper": "close",
                    "reasoning": "Reached final waypoint, grasping."}

        target_q = self.WAYPOINTS[self.waypoint_idx]
        current_q = np.array(state["joint_angles"])
        err = np.linalg.norm(np.array(target_q) - current_q)

        if err < 0.05:
            self.waypoint_idx += 1
            if self.waypoint_idx >= len(self.WAYPOINTS):
                return {"action": "grasp", "target_q": target_q, "gripper": "close",
                        "reasoning": "Final waypoint reached, grasping."}
            target_q = self.WAYPOINTS[self.waypoint_idx]

        # Compute delta toward target
        delta_q = (np.array(target_q) - current_q) * 0.15
        delta_q = np.clip(delta_q, -0.08, 0.08)

        return {"action": "move", "delta_q": delta_q.tolist(), "gripper": "open",
                "reasoning": f"Waypoint {self.waypoint_idx}/{len(self.WAYPOINTS)}, err={err:.3f}"}


class OpenAIVLM:
    """Real VLM via OpenAI GPT-4o. Set OPENAI_API_KEY env var."""

    def __init__(self, model="gpt-4o-mini", api_key=None, base_url=None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError("Set OPENAI_API_KEY environment variable.")
        try:
            import openai
            kwargs = {"api_key": self.api_key}
            if base_url:
                kwargs["base_url"] = base_url
            self.client = openai.OpenAI(**kwargs)
        except ImportError:
            raise ImportError("Install openai:  uv pip install openai")

    def _encode_image(self, rgb_array):
        img = Image.fromarray(rgb_array)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def decide(self, rgb_dict, state):
        eye_b64 = self._encode_image(rgb_dict["eye_in_hand"])
        env_b64 = self._encode_image(rgb_dict["env_camera"])
        system_prompt = (
            "You are a robot vision controller with two cameras: "
            "'env_camera' shows the whole scene, 'eye_in_hand' shows gripper view. "
            "Pick the red apple. Respond ONLY with JSON: "
            '{"action":"move|grasp|hold","delta_q":[j1,j2,j3,j4,j5,j6],"gripper":"open|close","reasoning":"..."}'
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Environment:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{env_b64}"}},
                    {"type": "text", "text": "Gripper view:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{eye_b64}"}},
                    {"type": "text", "text": f"End-effector: {state['end_effector_pos']}"},
                ],
            },
        ]
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0.2, max_tokens=300
        )
        text = response.choices[0].message.content
        try:
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            return json.loads(text.strip())
        except Exception as e:
            print(f"VLM parse error: {e}\nRaw: {text}")
            return {"action": "hold", "reasoning": "Parse error"}


# ============================================================================
# Robot Environment (Kinematic Control)
# ============================================================================
class QwenVLM(OpenAIVLM):
    """Alibaba Cloud Bailian Qwen VLM (direct joint controller with history).
    The LLM directly outputs per-joint increments (delta_q) based on current
    state and recent history, learning from past steps to refine motion.
    """

    def __init__(self, max_history=8):
        super().__init__(
            model=QWEN_CONFIG["model"],
            api_key=QWEN_CONFIG["api_key"],
            base_url=QWEN_CONFIG["base_url"],
        )
        self.max_history = max_history
        self.history = []
        self.step_count = 0
        self._last_state = None
        self._last_decision = None
        self._grasp_decided = False
        self._prev_dist = None
        self._stuck_count = 0

    def _record_step(self, current_state):
        """Append previous step to history using current state as the result."""
        if self._last_state is None or self._last_decision is None:
            return
        prev_jnt = self._last_state.get("joint_angles", [])
        curr_jnt = current_state.get("joint_angles", [])
        prev_ee = np.array(self._last_state.get("end_effector_pos", [0.0, 0.0, 0.0]))
        curr_ee = np.array(current_state.get("end_effector_pos", [0.0, 0.0, 0.0]))
        prev_apple = np.array(self._last_state.get("apple_pos", [0.0, 0.0, 0.0]))
        curr_apple = np.array(current_state.get("apple_pos", [0.0, 0.0, 0.0]))
        prev_dist = float(np.linalg.norm(prev_ee - prev_apple))
        curr_dist = float(np.linalg.norm(curr_ee - curr_apple))
        delta_ee = (curr_ee - prev_ee).tolist()

        limit_flags = []
        for i, (lo, hi) in enumerate(JOINT_LIMITS):
            j = float(prev_jnt[i])
            if j <= lo + 0.03:
                limit_flags.append(f"J{i+1}@LOW")
            elif j >= hi - 0.03:
                limit_flags.append(f"J{i+1}@HIGH")

        self.history.append({
            "step": self.step_count - 1,
            "joints": [round(float(j), 3) for j in prev_jnt],
            "delta_q": self._last_decision.get("delta_q", [0.0] * 6),
            "gripper": self._last_decision.get("gripper", "open"),
            "dist_before": round(prev_dist, 4),
            "dist_after": round(curr_dist, 4),
            "delta_ee": [round(float(v), 4) for v in delta_ee],
            "limits": limit_flags,
            "reasoning": self._last_decision.get("reasoning", "")[:60],
        })
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def _format_history(self):
        if not self.history:
            return "No history yet."
        lines = ["=== RECENT HISTORY (last {} steps) ===".format(len(self.history))]
        for h in self.history:
            improve = h["dist_before"] - h["dist_after"]
            sign = "good" if improve > 0.001 else ("bad" if improve < -0.001 else "neutral")
            limits_txt = f" limits={h['limits']}" if h.get("limits") else ""
            d_ee = h.get("delta_ee", [0, 0, 0])
            lines.append(
                f"Step {h['step']}: dq={h['delta_q']}, d_ee=[{d_ee[0]:.3f},{d_ee[1]:.3f},{d_ee[2]:.3f}],"
                f"{limits_txt} dist={h['dist_before']:.4f}->{h['dist_after']:.4f} ({sign})"
            )
        return "\n".join(lines)

    def _joint_effect_hint(self, jnt, ee, apple):
        """Generate qualitative task-space guidance based on current configuration."""
        dx = apple[0] - ee[0]
        dy = apple[1] - ee[1]
        dz = apple[2] - ee[2]
        lines = [
            "=== TASK-SPACE ERROR ===",
            f"Apple is at [{apple[0]:.3f}, {apple[1]:.3f}, {apple[2]:.3f}]",
            f"Gripper is at [{ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}]",
            f"You need to move: dx={dx:+.3f}m, dy={dy:+.3f}m, dz={dz:+.3f}m",
            "",
            "=== JOINT EFFECT GUIDE (for THIS configuration) ===",
        ]

        j2, j3, j5 = float(jnt[1]), float(jnt[2]), float(jnt[4])

        lines.append(f"J1 ({jnt[0]:.2f}): rotates base. Apple dy={dy:.3f}. Keep near 0 if aligned.")

        if j3 > 1.0:
            lines.append(f"J2 ({j2:.2f}): when J3 is large (+{j3:.1f}), INCREASING J2 raises Z, DECREASING J2 lowers Z.")
        else:
            lines.append(f"J2 ({j2:.2f}): higher value lifts arm forward/up.")

        if j3 >= 1.8:
            lines.append(f"J3 ({j3:.2f}): NEAR UPPER LIMIT. Cannot increase further. To extend X, use J2/J5 instead.")
        elif j3 > 0.5:
            lines.append(f"J3 ({j3:.2f}): positive extends arm. Increasing raises both X and Z.")
        else:
            lines.append(f"J3 ({j3:.2f}): increase toward +1.0~+1.6 to extend reach. AVOID negative.")

        lines.append(f"J4 ({jnt[3]:.2f}): wrist roll. Mainly affects orientation, not position.")

        if j3 > 1.0:
            lines.append(f"J5 ({j5:.2f}): when arm is extended, INCREASING J5 bends wrist UP (raises Z), DECREASING bends DOWN.")
        else:
            lines.append(f"J5 ({j5:.2f}): higher value bends wrist up.")

        lines.append(f"J6 ({jnt[5]:.2f}): wrist yaw. Mainly affects orientation, not position.")

        lines.append("")
        lines.append("=== STRATEGY ===")
        if dz < -0.05 and j3 >= 1.8:
            lines.append(f"Priority: LOWER Z by {abs(dz):.3f}m. J3 is maxed out, so DECREASE J2 and/or DECREASE J5.")
        elif dz > 0.05:
            lines.append(f"Priority: RAISE Z by {dz:.3f}m. INCREASE J2 and/or INCREASE J5.")
        elif dx > 0.05:
            lines.append(f"Priority: MOVE FORWARD (+X) by {dx:.3f}m. INCREASE J3 (if room) or adjust J2/J5.")
        elif dx < -0.05:
            lines.append(f"Priority: MOVE BACK (-X) by {abs(dx):.3f}m. DECREASE J3 or adjust J2.")
        else:
            lines.append("Near target. Fine-tune all joints for alignment.")

        return "\n".join(lines)

    def _stuck_warning(self, dist, ee, apple):
        if self._prev_dist is not None:
            if abs(dist - self._prev_dist) < 0.008:
                self._stuck_count += 1
            else:
                self._stuck_count = 0
        self._prev_dist = dist

        dx = apple[0] - ee[0]
        dz = apple[2] - ee[2]

        if self._stuck_count >= 4:
            advice = []
            if dz < -0.03:
                advice.append("LOWER Z: decrease J2 or decrease J5")
            elif dz > 0.03:
                advice.append("RAISE Z: increase J2 or increase J5")
            if dx > 0.03:
                advice.append("FORWARD: increase J3 (if not at limit)")
            elif dx < -0.03:
                advice.append("BACK: decrease J2")
            advice_str = "; ".join(advice) if advice else "try small adjustments to J1/J4/J6"

            return (
                f"\n=== WARNING: STUCK FOR {self._stuck_count} STEPS ===\n"
                f"Distance has not improved. IGNORE old successful patterns — they no longer apply. "
                f"Current need: dx={dx:+.3f}, dz={dz:+.3f}. {advice_str}.\n"
            )
        return ""

    def decide(self, rgb_dict, state):
        """Ask Qwen for direct joint increments, feeding history for in-context learning."""
        self._record_step(state)

        ee = np.array(state["end_effector_pos"])
        apple = np.array(state["apple_pos"])
        dist = float(np.linalg.norm(ee - apple))
        jnt = state["joint_angles"]
        grip = state["gripper_opening"]

        system_prompt = (
            "You are a robot-arm low-level controller. You directly output per-joint angle "
            "increments (delta_q) for a 6-DOF arm to pick a red apple. "
            "Study the provided history and joint-effect guide. Choose delta_q that moves the "
            "gripper toward the apple in task space (X, Y, Z). "
            "Respond ONLY with JSON: "
            '{"action":"move|grasp|hold","delta_q":[j1,j2,j3,j4,j5,j6],"'
            '"gripper":"open|close","reasoning":"..."}'
        )

        history_text = self._format_history()
        effect_hint = self._joint_effect_hint(jnt, ee, apple)
        stuck_text = self._stuck_warning(dist, ee, apple)

        user_text = (
            f"{history_text}\n\n"
            f"{effect_hint}\n"
            f"{stuck_text}\n"
            f"=== CURRENT STATE ===\n"
            f"- Step: {self.step_count}\n"
            f"- Joint angles: [{', '.join(f'{float(j):.3f}' for j in jnt)}] rad\n"
            f"- Joint limits: {', '.join(f'J{i+1}[{lo:.2f},{hi:.2f}]' for i,(lo,hi) in enumerate(JOINT_LIMITS))}\n"
            f"- Distance: {dist:.4f} m | Gripper: {grip:.4f} m | Phase: {state['phase']}\n\n"
            f"{state.get('vision_text', '')}\n\n"
            f"=== RULES ===\n"
            f"1. Output delta_q: 6 floats. Max ±0.12 rad (code clips; prefer ±0.08).\n"
            f"2. action='move' normally. 'grasp' when distance <= 0.05m. 'hold' after grasp.\n"
            f"3. gripper='open' during approach, 'close' when grasping.\n"
            f"4. Use the JOINT EFFECT GUIDE above — do not blindly repeat old history.\n"
            f"5. If a joint is at its limit, do NOT command further movement in that direction.\n"
            f"Output ONLY the JSON object, no markdown, no explanation outside JSON."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]

        decision = None
        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=0.2, max_tokens=300
            )
            text = response.choices[0].message.content
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            decision = json.loads(text.strip())
            action = decision.get("action", "move")
            reasoning = decision.get("reasoning", "")
        except Exception as e:
            print(f"Qwen parse error: {e}; falling back to proximity rule.")
            action = "grasp" if dist <= 0.05 else "move"
            reasoning = f"Fallback due to parse error: {e}"
            decision = {
                "action": action,
                "delta_q": [0.0] * 6,
                "gripper": "open" if dist > 0.05 else "close",
                "reasoning": reasoning,
            }

        # Safety: clamp delta_q
        raw_delta_q = decision.get("delta_q", [0.0] * 6)
        if not isinstance(raw_delta_q, list) or len(raw_delta_q) != 6:
            raw_delta_q = [0.0] * 6
        delta_q = np.array([float(v) for v in raw_delta_q])
        delta_q = np.clip(delta_q, -0.12, 0.12)

        gripper = decision.get("gripper", "open")
        if gripper not in ("open", "close"):
            gripper = "open"

        if action == "grasp" or self._grasp_decided:
            self._grasp_decided = True
            result = {
                "action": "grasp",
                "delta_q": delta_q.tolist(),
                "gripper": "close",
                "reasoning": f"[Qwen] {reasoning}",
            }
        elif action == "hold":
            result = {
                "action": "hold",
                "delta_q": [0.0] * 6,
                "gripper": "close",
                "reasoning": f"[Qwen] {reasoning}",
            }
        else:
            result = {
                "action": "move",
                "delta_q": delta_q.tolist(),
                "gripper": gripper,
                "reasoning": f"[Qwen] {reasoning} | dist={dist:.3f}",
            }

        # Store for next history record
        self._last_state = {
            "joint_angles": list(jnt),
            "end_effector_pos": list(ee),
            "apple_pos": list(apple),
        }
        self._last_decision = result.copy()
        self.step_count += 1

        return result


class RobotEnv:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data = mujoco.MjData(self.model)
        self.cam = CameraSystem(self.model, self.data)
        self.vision = VisionProcessor(self.model, self.data)

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
    parser.add_argument("--vlm", choices=["mock", "openai", "qwen"], default="mock")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--save-frames", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("VLM Robot Arm Controller")
    print("=" * 60)
    print(f"VLM: {args.vlm} | Viewer: {'OFF' if args.no_viewer else 'ON'}")
    print(f"Frames: {'ON' if args.save_frames else 'OFF'} | Steps: {args.steps}")
    print("=" * 60)

    env = RobotEnv()
    if args.vlm == "mock":
        vlm = MockVLM()
    elif args.vlm == "qwen":
        vlm = QwenVLM()
    else:
        vlm = OpenAIVLM()

    dt = 0.05
    step_idx = 0

    if args.no_viewer:
        _run(env, vlm, args, dt, step_idx, use_viewer=False)
    else:
        _run(env, vlm, args, dt, step_idx, use_viewer=True)

    env.close()
    print("\nDone.")


def _run(env, vlm, args, dt, step_idx, use_viewer):
    loop_fn = _loop_with_viewer if use_viewer else _loop_headless
    loop_fn(env, vlm, args, dt, step_idx)


def _loop_headless(env, vlm, args, dt, step_idx):
    while step_idx < args.steps and not env.grasped:
        t0 = time.time()
        _control_step(env, vlm, args, step_idx)
        env.step()
        _sleep_remain(t0, dt)
        step_idx += 1
    print(f"\nFinished: {step_idx} steps, phase={env.phase}")


def _loop_with_viewer(env, vlm, args, dt, step_idx):
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.cam.distance = 1.5
        viewer.cam.lookat[:] = [0.2, 0, 0.3]

        while viewer.is_running() and step_idx < args.steps and not env.grasped:
            t0 = time.time()
            _control_step(env, vlm, args, step_idx)
            env.step()
            viewer.sync()
            _sleep_remain(t0, dt)
            step_idx += 1

    print(f"\nFinished: {step_idx} steps, phase={env.phase}")


def _control_step(env, vlm, args, step_idx):
    frames = env.cam.capture_both(save=args.save_frames)
    ee = env.get_end_effector_pos()
    apple = env.get_apple_pos()

    vision_text = env.vision.format_observation(frames, ee)
    state = {
        "end_effector_pos": ee.tolist(),
        "apple_pos": apple.tolist(),
        "joint_angles": env.joint_angles.tolist(),
        "gripper_opening": env.gripper_opening,
        "phase": env.phase,
        "vision_text": vision_text,
    }
    print(state)
    dec = vlm.decide(frames, state)
    action = dec.get("action", "hold")

    print(f"[Step {step_idx:03d}] {action:6s} | {dec.get('reasoning', '')[:55]}")

    if action == "move":
        dq = np.array(dec.get("delta_q", [0]*6))
        env.joint_angles += dq
        for i in range(6):
            lo, hi = JOINT_LIMITS[i]
            env.joint_angles[i] = float(np.clip(env.joint_angles[i], lo, hi))
        env.phase = "approach"
    elif action == "grasp":
        env.gripper_opening = 0.005
        env.phase = "grasp"
        env.grasped = True
        env.attach_apple()
    elif action == "hold":
        env.phase = "done"
        env.grasped = True

    env.set_joints(env.joint_angles, env.gripper_opening)


def _sleep_remain(t0, target_dt):
    elapsed = time.time() - t0
    sleep = target_dt - elapsed
    if sleep > 0:
        time.sleep(sleep)


if __name__ == "__main__":
    main()
