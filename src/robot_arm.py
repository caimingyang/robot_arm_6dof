from cadgen import build123d as bd
from cadgen import step

# Dimensions in millimeters
BASE_DIAMETER = 80.0
BASE_HEIGHT = 30.0
SHOULDER_DIAMETER = 50.0
SHOULDER_LENGTH = 60.0
UPPER_ARM_WIDTH = 30.0
UPPER_ARM_HEIGHT = 25.0
UPPER_ARM_LENGTH = 150.0
ELBOW_DIAMETER = 40.0
ELBOW_LENGTH = 50.0
FOREARM_WIDTH = 25.0
FOREARM_HEIGHT = 20.0
FOREARM_LENGTH = 120.0
END_FLANGE_DIAMETER = 35.0
END_FLANGE_HEIGHT = 15.0


@step(out="../STEP/robot_arm.step")
def robot_arm():
    # Base
    base = bd.Cylinder(BASE_DIAMETER / 2, BASE_HEIGHT)
    base.label = "base"

    # Shoulder joint (cylinder on top of base)
    shoulder = bd.Cylinder(SHOULDER_DIAMETER / 2, SHOULDER_LENGTH)
    shoulder = shoulder.moved(bd.Location((0, 0, BASE_HEIGHT)))
    shoulder.label = "shoulder_joint"

    # Upper arm
    upper_arm = bd.Box(UPPER_ARM_WIDTH, UPPER_ARM_HEIGHT, UPPER_ARM_LENGTH)
    upper_arm = upper_arm.moved(
        bd.Location((0, 0, BASE_HEIGHT + SHOULDER_LENGTH + UPPER_ARM_LENGTH / 2))
    )
    upper_arm.label = "upper_arm"

    # Elbow joint
    elbow = bd.Cylinder(ELBOW_DIAMETER / 2, ELBOW_LENGTH)
    elbow = elbow.moved(
        bd.Location((0, 0, BASE_HEIGHT + SHOULDER_LENGTH + UPPER_ARM_LENGTH))
    )
    elbow.label = "elbow_joint"

    # Forearm
    forearm = bd.Box(FOREARM_WIDTH, FOREARM_HEIGHT, FOREARM_LENGTH)
    forearm = forearm.moved(
        bd.Location((0, 0, BASE_HEIGHT + SHOULDER_LENGTH + UPPER_ARM_LENGTH + ELBOW_LENGTH + FOREARM_LENGTH / 2))
    )
    forearm.label = "forearm"

    # End flange
    end_flange = bd.Cylinder(END_FLANGE_DIAMETER / 2, END_FLANGE_HEIGHT)
    end_flange = end_flange.moved(
        bd.Location((0, 0, BASE_HEIGHT + SHOULDER_LENGTH + UPPER_ARM_LENGTH + ELBOW_LENGTH + FOREARM_LENGTH))
    )
    end_flange.label = "end_flange"

    # Combine all parts into an assembly
    assembly = bd.Compound(
        children=[base, shoulder, upper_arm, elbow, forearm, end_flange]
    )
    assembly.label = "robot_arm"
    return assembly


if __name__ == "__main__":
    robot_arm()
