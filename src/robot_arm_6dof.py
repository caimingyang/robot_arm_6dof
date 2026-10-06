from cadgen import build123d as bd
from cadgen import step

# ===== Dimensions (mm) =====
# Base
BASE_D = 140.0
BASE_H = 35.0

# Joint 1 (Waist)
J1_D = 90.0
J1_H = 45.0

# Link 1 - Shoulder column
SHOULDER_W = 60.0
SHOULDER_D = 50.0
SHOULDER_H = 80.0

# Joint 2 (Shoulder pitch)
J2_D = 55.0
J2_W = 70.0

# Link 2 - Upper arm
UPPER_ARM_L = 220.0
UPPER_ARM_W = 45.0
UPPER_ARM_H = 40.0

# Joint 3 (Elbow pitch)
J3_D = 48.0
J3_W = 60.0

# Link 3 - Forearm
FOREARM_L = 180.0
FOREARM_W = 40.0
FOREARM_H = 35.0

# Joint 4 (Wrist roll)
J4_D = 40.0
J4_H = 35.0

# Link 4 - Wrist segment
WRIST_L = 70.0
WRIST_W = 35.0
WRIST_H = 30.0

# Joint 5 (Wrist pitch)
J5_D = 35.0
J5_H = 30.0

# Link 5 - End segment
END_L = 50.0
END_W = 30.0
END_H = 25.0

# Joint 6 (End roll)
J6_D = 30.0
J6_H = 25.0

# Gripper
GRIPPER_BASE_W = 50.0
GRIPPER_BASE_D = 40.0
GRIPPER_BASE_H = 20.0
FINGER_L = 60.0
FINGER_W = 12.0
FINGER_H = 20.0
FINGER_GAP = 80.0  # max opening

# Colors (using label for identification)
COLOR_BASE = "base"
COLOR_JOINT = "joint"
COLOR_LINK = "link"
COLOR_GRIPPER = "gripper"


def make_cylinder(d, h, label, x=0, y=0, z=0):
    cyl = bd.Cylinder(d / 2, h)
    cyl = cyl.moved(bd.Location((x, y, z + h / 2)))
    cyl.label = label
    return cyl


def make_box(w, d, h, label, x=0, y=0, z=0):
    box = bd.Box(w, d, h)
    box = box.moved(bd.Location((x, y, z + h / 2)))
    box.label = label
    return box


@step(out="../STEP/robot_arm_6dof.step")
def robot_arm_6dof():
    parts = []

    # ===== Base =====
    base = make_cylinder(BASE_D, BASE_H, "base", 0, 0, 0)
    parts.append(base)

    # ===== Joint 1 (Waist) =====
    z = BASE_H
    j1 = make_cylinder(J1_D, J1_H, "joint1", 0, 0, z)
    parts.append(j1)

    # ===== Link 1 (Shoulder column) =====
    z += J1_H
    shoulder = make_box(SHOULDER_W, SHOULDER_D, SHOULDER_H, "link1", 0, 0, z)
    parts.append(shoulder)

    # ===== Joint 2 (Shoulder pitch) =====
    z += SHOULDER_H
    j2 = make_box(J2_W, J2_D, J2_D, "joint2", 0, 0, z)
    parts.append(j2)

    # ===== Link 2 (Upper arm) =====
    # Upper arm extends along +Z from shoulder
    z += J2_D
    upper_arm = make_box(UPPER_ARM_W, UPPER_ARM_H, UPPER_ARM_L, "link2", 0, 0, z)
    parts.append(upper_arm)

    # ===== Joint 3 (Elbow pitch) =====
    z += UPPER_ARM_L
    j3 = make_box(J3_W, J3_D, J3_D, "joint3", 0, 0, z)
    parts.append(j3)

    # ===== Link 3 (Forearm) =====
    z += J3_D
    forearm = make_box(FOREARM_W, FOREARM_H, FOREARM_L, "link3", 0, 0, z)
    parts.append(forearm)

    # ===== Joint 4 (Wrist roll) =====
    z += FOREARM_L
    j4 = make_cylinder(J4_D, J4_H, "joint4", 0, 0, z)
    parts.append(j4)

    # ===== Link 4 (Wrist segment) =====
    z += J4_H
    wrist = make_box(WRIST_W, WRIST_H, WRIST_L, "link4", 0, 0, z)
    parts.append(wrist)

    # ===== Joint 5 (Wrist pitch) =====
    z += WRIST_L
    j5 = make_box(J5_D, J5_H, J5_D, "joint5", 0, 0, z)
    parts.append(j5)

    # ===== Link 5 (End segment) =====
    z += J5_D
    end_link = make_box(END_W, END_H, END_L, "link5", 0, 0, z)
    parts.append(end_link)

    # ===== Joint 6 (End roll) =====
    z += END_L
    j6 = make_cylinder(J6_D, J6_H, "joint6", 0, 0, z)
    parts.append(j6)

    # ===== Gripper =====
    z += J6_H
    gripper_base = make_box(
        GRIPPER_BASE_W, GRIPPER_BASE_D, GRIPPER_BASE_H, "gripper_base", 0, 0, z
    )
    parts.append(gripper_base)

    z += GRIPPER_BASE_H
    # Left finger
    left_finger = make_box(FINGER_W, FINGER_H, FINGER_L, "finger_left", -FINGER_GAP / 4, 0, z)
    parts.append(left_finger)

    # Right finger
    right_finger = make_box(FINGER_W, FINGER_H, FINGER_L, "finger_right", FINGER_GAP / 4, 0, z)
    parts.append(right_finger)

    # ===== Assemble =====
    assembly = bd.Compound(children=parts)
    assembly.label = "robot_arm_6dof"
    return assembly


if __name__ == "__main__":
    robot_arm_6dof()
