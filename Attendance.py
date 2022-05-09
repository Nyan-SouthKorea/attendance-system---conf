from pydoc import classname
from re import I
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import time
import z_face_auto_tilt as fat
import z_brightness_manage as bm

# images안에 list로 들어있는 사진들을 자동으로 인코딩 해주는 함수 제작.
def findEncodings(imgs):
    encodeList = []
    for img in imgs:
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# 학생의 출석체크한 시간과 이름을 기록한다. name의 이름과 함께 이 함수가 호출되면 csv파일에 이름 중복확인 후 이름과 시간을 기록한다.
def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0]) # entry[0]는 이름이다.
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

# 갤러리 안에 사진에서 얼굴만 따로 crop하여 저장해놓고, flip한 사진도 추가로 저장 해 놓는다.
def crop_images_dlib(imgs, face_tilt, face_flip):
    counter = 1
    crop_gallary = []
    for img in imgs:
        if face_tilt == True:
            img = fat.face_tilt(img)
        bbox_frame = []
        bbox_frame = face_recognition.face_locations(img, model = "hog")
        if len(bbox_frame) > 0:
            bbox_frame = bbox_frame[0]
            y1 = int(bbox_frame[0])
            x2 = int(bbox_frame[1])
            y2 = int(bbox_frame[2])
            x1 = int(bbox_frame[3])
            img = img[y1:y2,x1:x2]
            counter += 1
            crop_gallary.append(img)
    if face_flip == True:
        flip_imgs = []
        for img in crop_gallary:
            img = cv2.flip(img, 1)
            flip_imgs.append(img)

        for flip_img in flip_imgs:
            crop_gallary.append(flip_img)

    for img in crop_gallary:
        img4show = cv2.resize(img, (250, 250))
        cv2.imshow('Webcam', img4show)
        cv2.waitKey(1)

    return crop_gallary

def imgs_resize(imgs, h, w):
    new_imgs = []
    for img in imgs:
        img = cv2.resize(img, (w, h))
        new_imgs.append(img)
    return new_imgs

# =======================================================
# ==================옵 션 설 정===========================
# =======================================================
auto_rotate = True
flip_enable = True
bright_sync = False
img_size_sync = True
face_size = 310
# =======================================================
# =======================================================

# 자동으로 폴더에서 사진들을 불러올 때 경로 지정 후 폴더 내에 모든 파일들을 print하여 확인한다.
path = 'Gallery'
imgs = []
classNames = []
myList = os.listdir(path)

# images안에 사진들을 리스트로 집어넣고, classNames안에 사진들의 이름을 .jpg를 제외하고 집어넣는다.
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    imgs.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

# className에 flip된 이름 리스트를 넣어주는 과정. hardcopy를 해야 해서 불필요해 보이는 for문이 있다.
if flip_enable == True:
    classNames_flip = []
    for className in classNames:
        className = '{}_flip'.format(className)
        classNames_flip.append(className)
    for className_flip in classNames_flip:
        classNames.append(className_flip)
print(classNames)

# 갤러리 안 사진들을 얼굴만 crop하여 저장해 놓는다(여기서 flip사진들도 추가됨)
imgs = crop_images_dlib(imgs, auto_rotate, flip_enable)

# 얼굴의 사이즈를 모두 통일시켜서 비교한다.
if img_size_sync == True:
    imgs = imgs_resize(imgs, face_size, face_size)

# 이미지 내 사진들의 밝기를 평균으로 동일하게 맞추어 놓는다.
if bright_sync == True:
    br_list = []
    for img in imgs:
        br = bm.mean_v_magic(img)
        br_list.append(br)
        br_mean = int(sum(br_list) / len(br_list))
    br_done_imgs = []
    for img in imgs:
        img = bm.brightness_set(img, br_mean)
        br_done_imgs.append(img)
        img4show = cv2.resize(img, (250, 250))
        cv2.imshow('Webcam', img4show)
        cv2.waitKey(1)

    # encodeListKnown 이라는 list안에 path폴더 내 모든 사진들을 인코딩 해서 넣는다.
    encodeListKnown = findEncodings(br_done_imgs)
else:
    encodeListKnown = findEncodings(imgs)
print('사진 인코딩 완료!')

# 웹캠 설정을 해준다.
cap = cv2.VideoCapture(0)
# webcam에서 프레임을 하나씩 받기 위해 while True로 돌려준다.
while True:
    success, img = cap.read() # 이미지를 받는다.
    h, w, c = img.shape
    print(h, w)
    if bright_sync == True:
        img = bm.brightness_set(img, br_mean)
    if auto_rotate == True:
        img = fat.face_tilt(img)
    facesCurFrame = face_recognition.face_locations(img) # 이전처럼 끝에 [0] 안붙이면 웹캠 안에 모든 얼굴의 위치가 저장된다.(faces of current frame)
    cam_crop_list = []
    cam_crop = img
    for bbox in facesCurFrame:
        y1 = int(bbox[0])
        x2 = int(bbox[1])
        y2 = int(bbox[2])
        x1 = int(bbox[3])
        cam_crop = img[y1:y2,x1:x2]
        cam_crop_list.append(cam_crop)
        break # 얼굴 한장만 가져가기 위해 break를 한다.
    
    if img_size_sync == True:
        cam_crop = cv2.resize(cam_crop, (face_size, face_size))
        
    encodesCurFrame = face_recognition.face_encodings(cam_crop, known_face_locations=None, num_jitters=1, model="small") # webcam도 인코딩 해준다.(encodes of current frame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame): # 웹캠에서 인식된 1개 이상의 얼굴을 각자 encodeFace와 faceLoc안에 집어넣는다.
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace) # faceDis 안에서 가장 적은 Distance가 Best match가 될 것이다.
        matchIndex = np.argmin(faceDis) # 가장 적은 값을 반환해 준다. (거리가 가장 가깝다 = 가장 비슷한 인물이다)

        # 이제 webcam 얼굴에 b box쳐서 가장 비슷한 사람 이름을 써주어야 한다.
        if len(faceDis) > 0:
            if flip_enable == True:
                for cnt in range(int((len(classNames) / 2))):
                    cnt_2 = int(cnt + (len(classNames) / 2))
                    per_1 = round((1 - float(faceDis[cnt])), 2)
                    per_2 = round((1 - float(faceDis[cnt_2])), 2)
                    print('{:10}: {:4} / {:10}: {:4}'.format(classNames[cnt], per_1, classNames[cnt_2], per_2))
            else:
                for cnt in range(len(classNames)):
                    per_1 = round((1 - float(faceDis[cnt])), 2)
                    print('{:10}: {:4}'.format(classNames[cnt], per_1))
            print('=============================================')
            
            name = classNames[matchIndex]
            DetectPer = 1 - float(faceDis[matchIndex])
            DetectPer = round(DetectPer, 2)
            DetectPer = str(DetectPer)
            y1, x2, y2, x1 = faceLoc # 얼굴 b box 좌표를 받는다.
            y1, x2, y2, x1 = y1, x2, y2, x1
            cv2.rectangle(img,(x1, y1), (x2, y2), (0, 255, 0), 2) # 얼굴 b box친다.
            cv2.rectangle(img,(x1, y2), (x2, y2+35), (0, 255, 0),cv2.FILLED) # 글씨창 친다.
            cv2.putText(img, '{},{}%'.format(name,DetectPer) ,(x1+6, y2+17), cv2.FONT_HERSHEY_COMPLEX, 0.5, (30, 30, 30), 1) # 인식된 객체의 이름을 쓴다.
            markAttendance(name) # csv파일에 출석과 시간을 기록한다.

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)