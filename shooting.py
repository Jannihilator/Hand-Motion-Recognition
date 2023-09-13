import cv2
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import GREEN_COLOR, RED_COLOR
# import time
from utils import CvFpsCalc
import math




cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
mpHands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
face_detection=mp_face_detection.FaceDetection(min_detection_confidence=0.5)
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.5)
mpDraw = mp.solutions.drawing_utils
cvFpsCalc = CvFpsCalc(buffer_len=10)
lastX, lastY = 960, 0
counter=0
fire=False
hold=False
hit= False
while True:
    fps = cvFpsCalc.get()
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resultsF = face_detection.process(imgRGB)
    # Draw the face detection annotations on the imgRGB.
    results = hands.process(imgRGB)
    if resultsF.detections:
        for detection in resultsF.detections:
            mpDraw.draw_detection(img, detection)
            relative_bounding_box = detection.location_data.relative_bounding_box
            face1 = (round(relative_bounding_box.xmin*960), round(relative_bounding_box.ymin*540))
            face2 = (face1[0]+round(relative_bounding_box.width*960), face1[1]+round(relative_bounding_box.height*540))
            break
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # for id, lm in enumerate(handLms.landmark):
            w,h= 960, 540
            #cx, cy = int(lm.x*w), int(lm.y*h)
            refX, refY = int(
                handLms.landmark[8].x*w), int(handLms.landmark[8].y*h)
            inX, inY = int(
                handLms.landmark[10].x*w), int(handLms.landmark[10].y*h)
            outX, outY = int(
                handLms.landmark[11].x*w), int(handLms.landmark[11].y*h)
            aimX, aimY = int(
                handLms.landmark[12].x*w), int(handLms.landmark[12].y*h)
            # cv2.circle(img, (aimX, aimY), 10, RED_COLOR, -1)
            if not hold and not fire and ((lastX*refY-lastY*refX)/(math.sqrt(lastX**2+lastY**2)*math.sqrt(refX**2+refY**2)))<-0.03:
                fire=True
                diffX, diffY = int((outX-inX)), int((outY-inY))
                originX, originY = outX+diffX, outY+diffY
                startX,startY=originX,originY
                endX, endY = originX+diffX, originY+diffY  
            if(fire):
                cv2.line(img,(startX, startY), (endX, endY), (0, 0,255), 7)
                startX+=diffX
                startY+=diffY
                endX+=diffX
                endY+=diffY
                if resultsF.detections:
                    if(endX>face1[0] and endX<face2[0] and endY>face1[1] and endY<face2[1]):
                        cv2.putText(img, "HIT", (face1[0], face1[1]-20), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)
                        cv2.rectangle(img, face1, face2, RED_COLOR, 3)
                        hit=True
                if(startX<0 or startY<0 or startX>960 or startY>540):
                    fire=False
                    hold=False
                    hit=False
                else:
                    hold=True
                    if hit:
                        cv2.putText(img, "HIT", (face1[0], face1[1]-20), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)
            
            lastX, lastY = refX, refY

            
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    
    
    
    cv2.putText(img,str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("imgRGB", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
