import cv2
import numpy as np
import dlib
import time

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
i = 0

def get_gaze_ratio(eye_points, facial_landmarks):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)
    # cv2.polylines(frame, [left_eye_region], True, (0, 0, 255), 2)

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
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    if left_side_white == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white / right_side_white

    # time.sleep(0.5)
    return gaze_ratio

while i<100:
    _, frame = cap.read()

    #グレースケール化
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #ランドマーク
    faces = detector(gray)
    if(len(faces)==0):
        print("顔がカメラに移っていないです。")
    else:
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()

            landmarks = predictor(gray, face)

            gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
            gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
            gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2
            print('gaze_ratio_left_eye', gaze_ratio_left_eye)
            print('gaze_ratio_right_eye', gaze_ratio_right_eye)

        # 瞳のトリミング処理
        # 右目：[36,,37,39, 40]　左目：[42, 43, 45, 46]
        # Right eye
        r_x1,r_y1 = landmarks.part(36).x,landmarks.part(36).y
        r_x2,r_y2 = landmarks.part(37).x,landmarks.part(37).y
        r_x3,r_y3 = landmarks.part(39).x,landmarks.part(39).y
        r_x4,r_y4 = landmarks.part(40).x,landmarks.part(40).y
        # Left eye
        l_x1,l_y1 = landmarks.part(42).x,landmarks.part(42).y
        l_x2,l_y2 = landmarks.part(43).x,landmarks.part(43).y
        l_x3,l_y3 = landmarks.part(45).x,landmarks.part(45).y
        l_x4,l_y4 = landmarks.part(46).x,landmarks.part(46).y

        #　トリミング範囲補正
        trim_val = 2
        r_frame_trim = frame[r_y2-trim_val:r_y4+trim_val, r_x1:r_x3]
        l_frame_trim = frame[l_y2-trim_val:l_y4+trim_val, l_x1:l_x3]

        # 拡大処理（5倍）
        r_height,r_width = r_frame_trim.shape[0],r_frame_trim.shape[1]
        l_height,l_width = l_frame_trim.shape[0],l_frame_trim.shape[1]
        r_frame_trim_resize = cv2.resize(r_frame_trim , (int(r_width*7.0), int(r_height*7.0)))
        l_frame_trim_resize = cv2.resize(l_frame_trim , (int(l_width*7.0), int(l_height*7.0)))

        # グレースケール処理
        r_frame_gray = cv2.cvtColor(r_frame_trim_resize, cv2.COLOR_BGR2GRAY)
        l_frame_gray = cv2.cvtColor(l_frame_trim_resize, cv2.COLOR_BGR2GRAY)

        #平滑化（ぼかし）
        r_frame_gray = cv2.GaussianBlur(r_frame_gray,(7,7),0)
        l_frame_gray = cv2.GaussianBlur(l_frame_gray,(7,7),0)

        # 2値化処理
        thresh = 80
        maxval = 255
        e_th,r_frame_black_white = cv2.threshold(r_frame_gray,thresh,maxval,cv2.THRESH_BINARY_INV)
        l_th,l_frame_black_white = cv2.threshold(l_frame_gray,thresh,maxval,cv2.THRESH_BINARY_INV)

        #輪郭の表示
        r_eye_contours, _ = cv2.findContours(r_frame_black_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        r_eye_contours = sorted(r_eye_contours, key=lambda x: cv2.contourArea(x), reverse=True) #輪郭が一番大きい順に並べる

        if(len(r_eye_contours)==0):
            print("Right Blink")
        else:
            for cnt in r_eye_contours:
                (x, y, w, h) = cv2.boundingRect(cnt)
                # cv2.drawContours(r_frame_trim_resize, [cnt], -1, (0,0,255),3) #輪郭の表示
                # cv2.rectangle(r_frame_trim_resize, (x, y), ((x + w, y + h)), (255, 0, 0), 2)#矩形で表示
                cv2.circle(r_frame_trim_resize, (int(x+w/2), int(y+h/2)), int((w+h)/4), (255, 0, 0), 2) #円で表示
                cv2.circle(frame, (int(r_x1+(x+w)/10), int(r_y2-3+(y+h)/10)), int((w+h)/20), (0, 255, 0), 1)    #元画像に表示
                break

        l_eye_contours, _ = cv2.findContours(l_frame_black_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        l_eye_contours = sorted(l_eye_contours, key=lambda x: cv2.contourArea(x), reverse=True) #輪郭が一番大きい順に並べる
        if(len(l_eye_contours)==0):
            print("Left Blink")
        else:
            for cnt in l_eye_contours:
                (x, y, w, h) = cv2.boundingRect(cnt)
                # cv2.drawContours(l_frame_trim_resize, [cnt], -1, (0,0,255),3) #輪郭の表示
                # cv2.rectangle(l_frame_trim_resize, (x, y), ((x + w, y + h)), (255, 0, 0), 2)#矩形で表示
                cv2.circle(l_frame_trim_resize, (int(x+w/2), int(y+h/2)), int((w+h)/4), (255, 0, 0), 2) #円で表示
                cv2.circle(frame, (int(l_x1+(x+w)/10), int(l_y2-3+(y+h)/10)), int((w+h)/20), (0, 255, 0), 1)    #元画像に表示
                break


        #画像の表示
        cv2.imshow("frame",frame)

        cv2.imshow("right eye trim",r_frame_trim_resize)
        cv2.imshow("left eye trim",l_frame_trim_resize)

        cv2.imshow("right eye black white",r_frame_black_white)
        cv2.imshow("left eye black white",l_frame_black_white)

        #ウィンドウの配置変更
        cv2.moveWindow('frame', 200,0)
        cv2.moveWindow('right eye trim', 100,100)
        cv2.moveWindow('left eye trim', 240,100)
        cv2.moveWindow('right eye black white', 100,250)
        cv2.moveWindow('left eye black white', 240,250)

    key = cv2.waitKey(1)
    if key ==27:
        break

def get_gaze_ratio(eye_points, facial_landmarks):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)
    # cv2.polylines(frame, [left_eye_region], True, (0, 0, 255), 2)

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
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    if left_side_white == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white / right_side_white
    return gaze_ratio

cv2.destroyAllWindows()
