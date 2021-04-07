import  cv2
import  dlib
from math import hypot

cap=cv2.VideoCapture(0)


glass=cv2.imread("sunglass2.png")
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

while True:
    success,img=cap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=detector(gray)

    for face in faces:
        landmark=predictor(gray,face)

        top_head=(landmark.part(27).x,landmark.part(27).y)
        left_eye=(landmark.part(18).x,landmark.part(18).y)
        right_eye=(landmark.part(26).x,landmark.part(26).y)

        left_up = (landmark.part(44).x, landmark.part(44).y)
        left_down = (landmark.part(46).x, landmark.part(46).y)

        right_up = (landmark.part(37).x, landmark.part(37).y)
        right_down = (landmark.part(41).x, landmark.part(41).y)

        eye_width=int(hypot(left_eye[0]-right_eye[0],left_eye[1]-right_eye[1])*1.2)
        eye_hight=int(eye_width*0.4)

        '''cv2.circle(img,left_up,3,(0,255,0),-1)
        cv2.circle(img, left_down, 3, (0, 255, 0), -1)
        cv2.circle(img, right_up, 3, (0, 255, 0), -1)
        cv2.circle(img, right_down, 3, (0, 255, 0), -1)'''

        e1 = int(hypot(left_up[0] - left_down[0], left_up[1] - left_down[1]))
        e2 = int(hypot(right_up[0] - right_down[0], right_up[1] - right_down[1]))
        #print(e1,'and',e2)
        if e1 and e2 >6:
            top_left=(int(top_head[0]-eye_width/2),
                               int(top_head[1]-eye_hight/2))
            bottom_right=(int(top_head[0]+eye_width/2),
                           int(top_head[1]+eye_hight/2))

            '''cv2.rectangle(img,(int(top_head[0]-eye_width/2),
                               int(top_head[1]-eye_hight/2)),
                          (int(top_head[0]+eye_width/2),
                           int(top_head[1]+eye_hight/2)),(0,255,0),2)'''
            eye_resize = cv2.resize(glass, (eye_width, eye_hight))
            eye_area=(img[top_left[1]:top_left[1] +eye_hight,
                     top_left[0]:top_left[0]+eye_width])

            eye_gray=cv2.cvtColor(eye_resize,cv2.COLOR_BGR2GRAY)
            _,eye_mask=cv2.threshold(eye_gray,25,255,cv2.THRESH_BINARY_INV)

            new_eye=cv2.bitwise_and(eye_area,eye_area, mask=eye_mask)
            final=cv2.add(new_eye,eye_resize)

            img[top_left[1]:top_left[1] + eye_hight,
            top_left[0]:top_left[0] + eye_width]=final
    
    
            #cv2.imshow("final", final)
            #cv2.imshow("Video", img)
            cv2.imshow("Video", img)
            '''
            cv2.imshow('sunglass pic', eye_resize)
            cv2.imshow("area", eye_area)
    
            cv2.imshow("mask",eye_mask )'''
        else:
            cv2.imshow("Video", img)
    if cv2.waitKey(1)  & 0xFF ==ord("q"):
        break
