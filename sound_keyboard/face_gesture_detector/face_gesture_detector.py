from enum import Enum
from typing_extensions import Unpack

class EyeDirection(Enum):
    CENTER = 0 # とれる？とれたら
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

class EyeState(Enum):
    CLOSE = 0
    OPEN = 1

class MouseState(Enum):
    CLOSE = 0
    OPEN = 1

class Gestures:

    def __init__(self):
        self.eye_direction = None
        self.left_eye_state = None
        self.right_eye_state = None
        self.mouse_state = None


def detect_gestures(frame):
    """
    args:
    - frame: numpy.ndarray

    returns:
        Gestures
    """
    # frameはcap.read()の返り値です。
    # このフレームからジェスチャーを検出して、上のジェスチャーのEnumのリストを返していただけると助かります！
    
    pass
