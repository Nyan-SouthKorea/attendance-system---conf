import cv2
import mediapipe as mp
import numpy as np
import math
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


def get_head_tilt(img, draw_dot = False):
    try: 
        image = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []
        edited_img, left_xy, right_xy = image, [0, 0], [0, 0]
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        dot_x = lm.x * img_w
                        dot_y = lm.y * img_h
                        dot_x, dot_y = int(dot_x), int(dot_y)
                        if idx == 33:
                            dot_color = (255, 0, 0)
                            left_xy[0] = dot_x
                            left_xy[1] = dot_y
                        elif idx == 263:
                            dot_color = (0, 255, 0)
                            right_xy[0] = dot_x
                            right_xy[1] = dot_y
                        elif idx == 1:
                            dot_color = (0, 0, 0)
                        elif idx == 61:
                            dot_color = (0, 0, 0)
                        elif idx == 291:
                            dot_color = (0, 0, 0)
                        elif idx == 199:
                            dot_color = (0, 0, 0)

                        if draw_dot == True:
                            cv2.line(image, (dot_x, dot_y), (dot_x, dot_y), dot_color, 5)
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])       
                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)
                focal_length = 1 * img_w
                cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                        [0, focal_length, img_w / 2],
                                        [0, 0, 1]])
                dist_matrix = np.zeros((4, 1), dtype=np.float64)
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                rmat, jac = cv2.Rodrigues(rot_vec)
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360
                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y * 10) , int(nose_2d[1] - x * 10))
            edited_img = image
    except:
        edited_img, left_xy, right_xy = img, [0, 0], [0, 0]
    return edited_img, left_xy, right_xy


def rotate_img(img, angle, scale = 1):
    try:
        angle = -angle
        h, w, c = img.shape
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        result = cv2.warpAffine(img, matrix, (w, h))
    except:
        result = img
    return result

def face_tilt(img, draw_dot = False):
    try: 
        img, left_xy, right_xy = get_head_tilt(img, draw_dot = draw_dot)
        
        
        l_x, l_y, r_x, r_y = left_xy[0], left_xy[1], right_xy[0], right_xy[1]
        tri_x = r_x - l_x
        tri_y = r_y - l_y
        tri_theta = math.atan(tri_y / tri_x)
        tri_theta = tri_theta * 180 / math.pi
        # print(tri_theta)
        img = rotate_img(img, -tri_theta)
    except: 
        pass
    return img

# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#     success, img = cap.read()
#     img_1 = face_tilt(img, draw_dot = True)
#     img_2 = face_tilt(img, draw_dot = False)
#     cv2.imshow('test 1', img_1)
#     cv2.imshow('test 2', img_2)
#     cv2.waitKey(1)
