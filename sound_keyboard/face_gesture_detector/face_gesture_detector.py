from enum import Enum
import time
from sound_keyboard.queue import get_queue
from sound_keyboard.face_gesture_detector.face_detection import inference
import cv2
import dlib
import numpy as np
import sys


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
        self.debug = len(sys.argv) >= 2 and sys.argv[1] == 'DEBUG'

    def get_gaze_state(self, x):
        if x <= 0.54:
            return EyeDirection.LEFT
        if x >= 0.57:
            return EyeDirection.RIGHT
        return EyeDirection.CENTER


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

    def gaze_preprocess(self, frame, face):

        #ランドマーク
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        landmarks = self.predictor(gray, face)
        return landmarks, gray

    def get_eye_blink_state(self, frame, landmarks, facial_landmarks):
        x = [0 for i in range(len(facial_landmarks))]
        y = [0 for i in range(len(facial_landmarks))]

        for i, facial_landmark in enumerate(facial_landmarks):
            x[i] = landmarks.part(facial_landmark).x
            y[i] = landmarks.part(facial_landmark).y

        trim_val = 2
        frame_trim = frame[y[1]-trim_val:y[3]+trim_val,x[0]:x[2]]
        height, width = frame_trim.shape[0],frame_trim.shape[1]
        frame_trim_resize = cv2.resize(frame_trim , (int(width*7.0), int(height*7.0)))
        if self.debug:
            cv2.imshow("eye trim",frame_trim_resize)
        # gray scale
        frame_gray = cv2.cvtColor(frame_trim_resize, cv2.COLOR_BGR2GRAY)
        # 平滑化
        frame_gray = cv2.GaussianBlur(frame_gray,(7,7),0)
        # 二値化
        thresh = 80
        maxval = 255
        e_th, frame_black_white = cv2.threshold(frame_gray,thresh,maxval,cv2.THRESH_BINARY_INV)
        eye_contours, _ = cv2.findContours(frame_black_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(eye_contours) == 0:
            return EyeState.CLOSE
        # (rx, ry, rw, rh) = cv2.boundingRect(eye_contours[0])
        # cv2.circle(frame_trim_resize, (int(rx+rw/2), int(ry+rh/2)), int((rw+rh)/4), (255, 0, 0), 2) #円で表示
        # cv2.circle(frame, (int(x[0]+(rx+rw)/10), int(y[1]-3+(ry+rh)/10)), int((rw+rh)/20), (0, 255, 0), 1)    #元画像に表示
        return EyeState.OPEN

    def run(self):
        # この関数のwhile True:下でcapから画像を取得して、ジェスチャーを判別、
        # queueに(Gestureオブジェクト、time.time())の形でジェスチャーを入れていってください。
        while True:
            _, frame = self.cap.read()
            face = inference(frame)
            start, end = face
            if start == -1 and end == -1:
                cv2.imshow('frame', frame)
                cv2.waitKey(1)
                continue
        
            (left, top), (right, bottom) = start, end
            if self.debug:
                cv2.rectangle(frame, start, end, (255, 0, 0), 2)
            dlib_face = dlib.rectangle(left=left, top=top, right=right, bottom=bottom)
            landmarks, gray = self.gaze_preprocess(frame, dlib_face)
            if landmarks == -1 and gray == -1:
                continue
            left_facial_landmarks = [42, 43, 44, 45, 46, 47]
            right_facial_landmarks = [36, 37, 38, 39, 40, 41]
            left_gaze_right_level, left_white_space = self.get_gaze_right_level(
                left_facial_landmarks,
                landmarks,
                frame,
                gray
            )
            right_gaze_right_level, right_white_space = self.get_gaze_right_level(
                right_facial_landmarks,
                landmarks,
                frame,
                gray
            )
            gaze_right_level = (left_gaze_right_level + right_gaze_right_level) / 2
            print(gaze_right_level)
            left_blink_state = self.get_eye_blink_state(frame, landmarks, [42, 43, 45, 46])
            right_blink_state = self.get_eye_blink_state(frame, landmarks, [36, 37, 39, 40])
            print('right_blink_state', right_blink_state)
            print('left_blink_state', left_blink_state)
            if self.debug:
                cv2.imshow("frame", frame)
                key = cv2.waitKey(1)
                if key ==27:
                    break

            if self.queue.full():
                self.queue.queue.clear()

            #TODO(inazuma110, nadeemishikawa) 適当な値ではなく、ちゃんと画像からの検知に基づいてプロパティをセットしたGesturesクラスのオブジェクトを入れる
            self.queue.put((Gestures(
                eye_direction=self.get_gaze_state(gaze_right_level),
                left_eye_state=left_blink_state,
                right_eye_state=right_blink_state,
                mouth_state=MouthState.CLOSE,
            ), time.time()))

if __name__ == '__main__':
    queue = get_queue()
    FaceGestureDetector(queue).run()
