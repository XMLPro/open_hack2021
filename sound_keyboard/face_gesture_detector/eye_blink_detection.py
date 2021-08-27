import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import torch
import os
from PIL import Image
from torchvision import transforms
import cv2
from sound_keyboard.face_gesture_detector.enums import EyeState
from keras.models import load_model

model = load_model('./sound_keyboard/face_gesture_detector/2018_12_17_22_58_35.h5')


def get_eye_position(landmarks, eye_points):
    eye_region = np.array([
    (
        landmarks.part(eye_points[0]).x,
        landmarks.part(eye_points[0]).y
    ),
    (
        landmarks.part(eye_points[1]).x,
        landmarks.part(eye_points[1]).y
    ),
    (
        landmarks.part(eye_points[2]).x,
        landmarks.part(eye_points[2]).y
    ),
    (
        landmarks.part(eye_points[3]).x,
        landmarks.part(eye_points[3]).y
    ),
    (
        landmarks.part(eye_points[4]).x,
        landmarks.part(eye_points[4]).y
    ),
    (
        landmarks.part(eye_points[5]).x,
        landmarks.part(eye_points[5]).y
    )], np.int32)

    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])

    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2 - 10 # 目をつぶるとセンターの位置が若干上がるので調整
    distance_w = abs(max_x - min_x) + 30
    distance_h = distance_w * 26 // 34
    
    left = center_x - distance_w // 2
    top = center_y - distance_h // 2
    right = center_x + distance_w // 2
    bottom = center_y + distance_h // 2

    return left, top, right, bottom

def get_area(frame, position):
    left, top, right, bottom = position

    return frame[top:bottom,left:right]

def infer(image, type):
    pred = model.predict(image)
    
    return EyeState.OPEN if pred[0][0] > 0.05 else EyeState.CLOSE



def get_eye_state(frame, landmarks):

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    left_eye_position = get_eye_position(landmarks, [42, 43, 44, 45, 46, 47])
    right_eye_position = get_eye_position(landmarks, [36, 37, 38, 39, 40, 41])
    
    left_eye = get_area(frame, left_eye_position)
    right_eye = get_area(frame, right_eye_position)

    left_eye = cv2.resize(left_eye, (34, 26))
    right_eye = cv2.resize(right_eye, (34, 26))
    right_eye = cv2.flip(right_eye, flipCode=1)

    cv2.imshow('left_eye', left_eye)
    cv2.imshow('right_eye', right_eye)
    #print("left eye shape", left_eye.shape)
    #print("right eye shape", right_eye.shape)

    left_eye_input = left_eye.copy().reshape((1, 26, 34, 1)).astype(np.float32) / 255
    right_eye_input = right_eye.copy().reshape((1, 26, 34, 1)).astype(np.float32) / 255
    
    left_state = infer(left_eye_input, 'left')
    right_state = infer(right_eye_input, 'right')

    return left_state, right_state, left_eye_position, right_eye_position