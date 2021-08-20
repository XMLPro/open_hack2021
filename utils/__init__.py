from sound_keyboard.keyboard.state_controller import (
    Direction
)

from sound_keyboard.face_gesture_detector.face_gesture_detector import (
    EyeDirection
)

def convert_direction_to_eye_direction(direction):
    if direction == Direction.CENTER:
        return EyeDirection.CENTER
    elif direction == Direction.LEFT:
        return EyeDirection.LEFT
    elif direction == Direction.UP:
        return EyeDirection.UP
    elif direction == Direction.RIGHT:
        return EyeDirection.RIGHT
    elif direction == Direction.DOWN:
        return EyeDirection.DOWN
    else:
        raise Exception('invalid mapping')

def convert_eye_direction_to_direction(eye_direction):
    if eye_direction == EyeDirection.CENTER:
        return Direction.CENTER
    elif eye_direction == EyeDirection.LEFT:
        return Direction.LEFT
    elif eye_direction == EyeDirection.UP:
        return Direction.UP
    elif eye_direction == EyeDirection.RIGHT:
        return Direction.RIGHT
    elif eye_direction == EyeDirection.DOWN:
        return Direction.DOWN
    else:
        raise Exception('invalid mapping')