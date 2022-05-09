from pydoc import classname
from re import I
from zipfile import LargeZipFile
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import time
from threading import Thread
import torch
import pandas

def mean_v_magic(image):
    # 사진을 넣으면 hsv의 v값만 추출하여 그 사진의 명도 평균값을 출력한다.
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(img_hsv) # 3차원의 이미지를 1차원 3개로 나누어 준다.
    mean_v = []
    for list_v in v:
        list_v = float(sum(list_v) / len(list_v))
        mean_v.append(list_v)
    mean_v = int(sum(mean_v) / len(mean_v))
    return mean_v

def brightness_Cam2Gallary(cam, gallary):
    # 무조건 webcam 밝기를 프로필 사진에 맞춘다.
    cam_v = mean_v_magic(cam)
    gallary_v = mean_v_magic(gallary)
    if cam_v < gallary_v:
        print('더 밝음: 갤러리')
        value = gallary_v - cam_v
        array = np.full(cam.shape, (value, value, value), dtype = np.uint8)
        cam = cv2.add(cam, array)
    elif cam_v > gallary_v:
        print('더 밝음: 캠')
        value = cam_v - gallary_v
        array = np.full(cam.shape, (value, value, value), dtype = np.uint8)
        cam = cv2.subtract(cam, array)
    print('cam: {} / gallary: {}'.format(cam_v, gallary_v))
    return cam, gallary

def brightness_set(cam, brightness):
    value = 0
    cam_v = mean_v_magic(cam)
    if cam_v < brightness:
        value = brightness - cam_v
        array = np.full(cam.shape, (value, value, value), dtype = np.uint8)
        cam = cv2.add(cam, array)
    elif cam_v > brightness:
        value = cam_v - brightness
        array = np.full(cam.shape, (value, value, value), dtype = np.uint8)
        cam = cv2.subtract(cam, array)
    return cam