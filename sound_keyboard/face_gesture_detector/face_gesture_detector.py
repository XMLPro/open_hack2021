from enum import Enum
import time
from sound_keyboard.queue import get_queue

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


class FaceGestureDetector:

    def __init__(self, queue):
        self.queue = queue
    
    def run(self):
        # この関数のwhile True:下でcapから画像を取得して、ジェスチャーを判別、
        # queueに(Gestureオブジェクト、time.time())の形でジェスチャーを入れていってください。

        while True:

            if self.queue.full():
                self.queue.queue.clear()

            #TODO(inazuma110, nadeemishikawa) 適当な値ではなく、ちゃんと画像からの検知に基づいてプロパティをセットしたGesturesクラスのオブジェクトを入れる
            self.queue.put((Gestures(
                eye_direction=EyeDirection.CENTER,
                left_eye_state=EyeState.OPEN,
                right_eye_state=EyeState.OPEN,
                mouth_state=MouthState.CLOSE,
            ), time.time()))

if __name__ == '__main__':
    queue = get_queue()
    FaceGestureDetector(queue).run()