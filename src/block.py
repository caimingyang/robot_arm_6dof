from cadgen import build123d as bd
from cadgen import step

WIDTH = 100.0
DEPTH = 200.0
HEIGHT = 30.0


@step(out="../STEP/block.step")
def block():
    body = bd.Box(WIDTH, DEPTH, HEIGHT)
    body = body.moved(bd.Location((0, 0, HEIGHT / 2)))
    body.label = "block"
    return body


if __name__ == "__main__":
    block()
