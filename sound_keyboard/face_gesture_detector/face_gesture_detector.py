from enum import Enum
import time
from sound_keyboard.queue import get_queue
import cv2
import dlib
import numpy as np
from math import hypot

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
        self.cap = cv2.VideoCapture(0)
        self.queue = queue
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("./sound_keyboard/face_gesture_detector/shape_predictor_68_face_landmarks.dat")

    def get_gaze_state(self, x):
        if x <= 0.4:
            return EyeDirection.LEFT
        if x >= 0.7:
            return EyeDirection.RIGHT
        return EyeDirection.CENTER

    def get_mouth_state(self, mouth_ratio):
        if mouth_ratio > 6.8:
            return MouthState.CLOSE
        else:
            return MouthState.OPEN


    #ランドマークの重心
    def midpoint(self, p1, p2):
        return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

    #口開閉検知
    def get_mouth_ratio(self, eye_points, facial_landmarks):
        left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
        right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
        center_top = self.midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
        center_bottom = self.midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

        # hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
        # ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

        hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
        ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

        ratio = hor_line_lenght / ver_line_lenght
        return ratio

    # 視線検知
    def get_gaze_right_level(self, eye_points, facial_landmarks, frame, gray):
        left_eye_region = np.array([
            (
                facial_landmarks.part(eye_points[0]).x,
                facial_landmarks.part(eye_points[0]).y
            ),
            (
                facial_landmarks.part(eye_points[1]).x,
                facial_landmarks.part(eye_points[1]).y
            ),
            (
                facial_landmarks.part(eye_points[2]).x,
                facial_landmarks.part(eye_points[2]).y
            ),
            (
                facial_landmarks.part(eye_points[3]).x,
                facial_landmarks.part(eye_points[3]).y
            ),
            (
                facial_landmarks.part(eye_points[4]).x,
                facial_landmarks.part(eye_points[4]).y
            ),
            (
                facial_landmarks.part(eye_points[5]).x,
                facial_landmarks.part(eye_points[5]).y
            )], np.int32)

        height, width, _ = frame.shape
        mask = np.zeros((height, width), np.uint8)
        cv2.polylines(mask, [left_eye_region], True, 255, 2)
        cv2.fillPoly(mask, [left_eye_region], 255)
        eye = cv2.bitwise_and(gray, gray, mask=mask)

        min_x = np.min(left_eye_region[:, 0])
        max_x = np.max(left_eye_region[:, 0])
        min_y = np.min(left_eye_region[:, 1])
        max_y = np.max(left_eye_region[:, 1])

        gray_eye = eye[min_y: max_y, min_x: max_x]
        _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
        height, width = threshold_eye.shape
        left_side_threshold = threshold_eye[0: height, int(width / 2): width]
        left_side_white = cv2.countNonZero(left_side_threshold)

        right_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
        right_side_white = cv2.countNonZero(right_side_threshold)

        under_side_threshold = threshold_eye[int(height/2):height, 0:width]
        under_side_white = cv2.countNonZero(under_side_threshold)


        if left_side_white == 0:
            gaze_right_level = 0
        elif right_side_white == 0:
            gaze_right_level = 1
        else:
            gaze_right_level = left_side_white / (right_side_white+left_side_white)

        return gaze_right_level, under_side_white

    def preprocess(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #ランドマーク
        faces = self.detector(gray)
        if(len(faces)==0):
            print("顔がカメラに移っていないです。")
            return -1, -1
        face = faces[0]
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        landmarks = self.predictor(gray, face)
        return landmarks, gray

    def run(self):
        # この関数のwhile True:下でcapから画像を取得して、ジェスチャーを判別、
        # queueに(Gestureオブジェクト、time.time())の形でジェスチャーを入れていってください。
        while True:
            _, frame = self.cap.read()
            landmarks, gray = self.preprocess(frame)
            if landmarks == -1 and gray == -1:
                continue
            left_facial_landmarks = [36, 37, 38, 39, 40, 41]
            left_gaze_right_level, left_white_space = self.get_gaze_right_level(
                left_facial_landmarks,
                landmarks,
                frame,
                gray
            )
            right_facial_landmarks = [42, 43, 44, 45, 46, 47]
            right_gaze_right_level, right_white_space = self.get_gaze_right_level(
                right_facial_landmarks,
                landmarks,
                frame,
                gray
            )
            #口の開閉度測定
            faces = self.detector(gray)
            for face in faces:
                mouth_landmarks = [60, 61, 63, 64, 65, 67, 68]
                mouth_ratio = self.get_mouth_ratio(mouth_landmarks, landmarks)
                print(f"mouth_ratio : {mouth_ratio}")



            gaze_right_level = (left_gaze_right_level + right_gaze_right_level) / 2
            print(gaze_right_level)

            if self.queue.full():
                self.queue.queue.clear()

            #TODO(inazuma110, nadeemishikawa) 適当な値ではなく、ちゃんと画像からの検知に基づいてプロパティをセットしたGesturesクラスのオブジェクトを入れる
            self.queue.put((Gestures(
                eye_direction=self.get_gaze_state(gaze_right_level),
                left_eye_state=EyeState.OPEN,
                right_eye_state=EyeState.OPEN,
                mouth_state=self.get_mouth_state(mouth_ratio),
            ), time.time()))

if __name__ == '__main__':
    queue = get_queue()
    FaceGestureDetector(queue).run()
