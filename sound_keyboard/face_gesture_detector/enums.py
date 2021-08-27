from enum import Enum

class EyeDirection(Enum):
    CENTER = 0 # とれる？とれたら
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

class EyeState(Enum):
    CLOSE = 0
    OPEN = 1

class MouthState(Enum):
    CLOSE = 0
    OPEN = 1

class Gestures:
    def __init__(self, eye_direction = None, left_eye_state = None, right_eye_state = None, mouth_state = None):
        self.eye_direction = eye_direction
        self.left_eye_state = left_eye_state
        self.right_eye_state = right_eye_state
        self.mouth_state = mouth_state
    
    def __str__(self):
        return (
            "eye_direction: " + str(self.eye_direction) + "\n" +
            "left_eye_state: " + str(self.left_eye_state) + "\n" +
            "right_eye_state: " + str(self.right_eye_state) + "\n" +
            "mouth_state: " + str(self.mouth_state) + "\n"
        )