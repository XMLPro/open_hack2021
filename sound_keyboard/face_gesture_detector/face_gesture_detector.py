import time
from sound_keyboard.queue import get_queue
from sound_keyboard.face_gesture_detector.face_detection import inference
from sound_keyboard.face_gesture_detector.eye_blink_detection import get_eye_state
from sound_keyboard.face_gesture_detector.enums import (
    EyeDirection,
    EyeState,
    MouthState,
    Gestures
)
import copy
import cv2
import dlib
import numpy as np
from math import hypot
from sound_keyboard.face_gesture_detector.gaze_tracking import GazeTracking



class FaceGestureDetector:
    def __init__(self, queue):
        self.cap = cv2.VideoCapture(0)
        self.queue = queue
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("./sound_keyboard/face_gesture_detector/shape_predictor_68_face_landmarks.dat")
        # self.debug = len(sys.argv) >= 2 and sys.argv[1] == 'DEBUG'
        self.debug = True

        self.previous = None

        self.gaze = GazeTracking()

    def get_gaze_state(self, x):
        if x <= 0.54:
            return EyeDirection.LEFT
        if x >= 0.57:
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
        mouth_region = np.array([
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
        left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
        right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
        center_top = self.midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
        center_bottom = self.midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

        # hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
        # ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

        hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
        ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

        if ver_line_length == 0:
            ver_line_length += 0.001

        ratio = hor_line_length / ver_line_length
        return ratio, mouth_region

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

        return gaze_right_level, under_side_white, left_eye_region

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
            self.gaze.refresh(frame)
            frame = self.gaze.annotated_frame()
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

            # まばたきの計測
            left_eye_state, right_eye_state, left_eye_position, right_eye_position = get_eye_state(frame, landmarks)


            #口の開閉度測定
            mouth_landmarks = [60, 61, 63, 64, 65, 67]
            mouth_ratio, mouth_region = self.get_mouth_ratio(mouth_landmarks, landmarks)

            eye_direction = EyeDirection.CENTER
            if self.gaze.is_right():
                eye_direction = EyeDirection.RIGHT
            elif self.gaze.is_left():
                eye_direction = EyeDirection.LEFT
            else:
                eye_direction = EyeDirection.CENTER
            
            if self.debug:
                def draw_eye(position, state):
                    x, y, x1, y1 = position
                    cv2.rectangle(frame, (x, y), (x1, y1), (255, 0, 0), 2)

                    text = ""
                    if state == EyeState.OPEN:
                        text = "open"
                    else:
                        text = "close"

                    cv2.putText(frame, text, (x, y1), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 5)
                
                draw_eye(left_eye_position, left_eye_state)
                draw_eye(right_eye_position, right_eye_state)
                cv2.polylines(frame, pts=[mouth_region], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.imshow("frame", frame)
                key = cv2.waitKey(1)
                if key ==27:
                    break

            if self.queue.full():
                self.queue.queue.clear()

            gestures = Gestures(
                eye_direction=eye_direction,
                left_eye_state=left_eye_state,
                right_eye_state=right_eye_state,
                mouth_state=self.get_mouth_state(mouth_ratio),
            )

            if (
                    gestures.eye_direction != EyeDirection.CENTER or
                    gestures.left_eye_state != gestures.right_eye_state or
                    ((self.previous == None or self.previous.mouth_state == MouthState.CLOSE) and gestures.mouth_state == MouthState.OPEN)
                ):
                self.queue.put((gestures, time.time()))

            self.previous = copy.deepcopy(gestures)

if __name__ == '__main__':
    queue = get_queue()
    FaceGestureDetector(queue).run()
