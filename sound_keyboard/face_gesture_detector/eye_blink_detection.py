import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import torch
import os
from PIL import Image
from torchvision import transforms
import cv2
from sound_keyboard.face_gesture_detector.enums import EyeState

class _BaseWrapper(object):
    """
    Please modify forward() and backward() according to your task.
    """

    def __init__(self, model):
        super(_BaseWrapper, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.logits).to(self.device)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def forward(self, image):
        """
        Simple classification
        """
        self.model.zero_grad()
        self.logits = self.model(image)
        self.probs = F.softmax(self.logits, dim=1)
        return self.probs.sort(dim=1, descending=True)

    def backward(self, ids):
        """
        Class-specific backpropagation

        Either way works:
        1. self.logits.backward(gradient=one_hot, retain_graph=True)
        2. (self.logits * one_hot).sum().backward(retain_graph=True)
        """

        one_hot = self._encode_one_hot(ids)
        self.logits.backward(gradient=one_hot, retain_graph=True)

    def generate(self):
        raise NotImplementedError

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()


class BackPropagation(_BaseWrapper):
    def forward(self, image):
        self.image = image.requires_grad_()
        return super(BackPropagation, self).forward(self.image)

    def generate(self):
        gradient = self.image.grad.clone()
        self.image.grad.zero_()
        return gradient


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                                   bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ResidualBlock(nn.Module):

    def __init__(self, in_channeld, out_channels):
        super(ResidualBlock, self).__init__()

        self.residual_conv = nn.Conv2d(in_channels=in_channeld, out_channels=out_channels, kernel_size=1, stride=2,
                                       bias=False)
        self.residual_bn = nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3)

        self.sepConv1 = SeparableConv2d(in_channels=in_channeld, out_channels=out_channels, kernel_size=3, bias=False,
                                        padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3)
        self.relu = nn.ReLU()

        self.sepConv2 = SeparableConv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, bias=False,
                                        padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3)
        self.maxp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        res = self.residual_conv(x)
        res = self.residual_bn(res)
        x = self.sepConv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.sepConv2(x)
        x = self.bn2(x)
        x = self.maxp(x)
        return res + x


class Model(nn.Module):

    def __init__(self, num_classes):
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8, affine=True, momentum=0.99, eps=1e-3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(8, momentum=0.99, eps=1e-3)
        self.relu2 = nn.ReLU()

        self.module1 = ResidualBlock(in_channeld=8, out_channels=16)
        self.module2 = ResidualBlock(in_channeld=16, out_channels=32)
        self.module3 = ResidualBlock(in_channeld=32, out_channels=64)
        self.module4 = ResidualBlock(in_channeld=64, out_channels=128)

        self.last_conv = nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=3, padding=1)
        self.avgp = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, input):
        x = input
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.module1(x)
        x = self.module2(x)
        x = self.module3(x)
        x = self.module4(x)
        x = self.last_conv(x)
        x = self.avgp(x)
        x = x.view((x.shape[0], -1))
        return x



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
    center_y = (min_y + max_y) // 2 - 10
    distance = max(abs(max_x - min_x), abs(max_y - min_y)) + 30
    
    left = center_x - distance // 2
    top = center_y - distance // 2
    right = center_x + distance // 2
    bottom = center_y + distance // 2

    return left, top, right, bottom

def get_area(frame, position):
    left, top, right, bottom = position

    return frame[top:bottom,left:right]

def infer(image, net, type):
    img = torch.stack([image])
    bp = BackPropagation(model=net)
    _, ids = bp.forward(img)
    
    return ids[0][0]

def num2State(num):
    if num == 0:
        return EyeState.CLOSE
    elif num == 1:
        return EyeState.OPEN



def get_eye_state(frame, landmarks):

    left_eye_position = get_eye_position(landmarks, [42, 43, 44, 45, 46, 47])
    right_eye_position = get_eye_position(landmarks, [36, 37, 38, 39, 40, 41])
    
    left_eye = get_area(frame, left_eye_position)
    right_eye = get_area(frame, right_eye_position)

    left_eye = cv2.resize(left_eye, (24, 24))
    right_eye = cv2.resize(right_eye, (24, 24))

    cv2.imshow('left_eye', left_eye)
    cv2.imshow('right_eye', right_eye)

    # transform left_eye, right_eye to Tensor
    transformer = transforms.Compose([
        transforms.ToTensor()
    ])
    left_eye = transformer(Image.fromarray(left_eye).convert('L'))
    right_eye = transformer(Image.fromarray(right_eye).convert('L'))

    net = Model(num_classes=2)
    checkpoint = torch.load('./sound_keyboard/face_gesture_detector/model_11_96_0.1256.t7')
    net.load_state_dict(checkpoint['net'])
    net.eval()

    left_state = num2State(infer(left_eye, net, 'left'))
    right_state = num2State(infer(right_eye, net, 'right'))

    return left_state, right_state, left_eye_position, right_eye_position