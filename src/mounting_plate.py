from cadgen import build123d as bd
from cadgen import step

# Dimensions in millimeters
WIDTH = 100.0
DEPTH = 60.0
THICKNESS = 20.0
HOLE_D = 6.0  # M6 through-hole
HOLE_OFFSET_X = 10.0  # distance from edge to hole center
HOLE_OFFSET_Y = 10.0
CHAMFER_SIZE = 2.0

# Derived
HALF_W = WIDTH / 2
HALF_D = DEPTH / 2
HOLE_CX = HALF_W - HOLE_OFFSET_X
HOLE_CY = HALF_D - HOLE_OFFSET_Y


@step(out="../STEP/mounting_plate.step")
def mounting_plate():
    # Main block centered at origin, then shifted so bottom sits at z=0
    body = bd.Box(WIDTH, DEPTH, THICKNESS)
    body = body.moved(bd.Location((0, 0, THICKNESS / 2)))
    body.label = "mounting_plate"

    # Four M6 through-holes at corners
    holes = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            hole = bd.Cylinder(HOLE_D / 2, THICKNESS + 1)
            hole = hole.moved(bd.Location((sx * HOLE_CX, sy * HOLE_CY, THICKNESS / 2)))
            hole.label = f"hole_{sx}_{sy}"
            holes.append(hole)

    # Subtract holes
    part = body
    for hole in holes:
        part = part - hole

    # Chamfer top edges (z = THICKNESS face)
    top_edges = part.edges().group_by(bd.Axis.Z)[-1]
    part = part.chamfer(length=CHAMFER_SIZE, length2=CHAMFER_SIZE, edge_list=top_edges)

    return part


if __name__ == "__main__":
    mounting_plate()
