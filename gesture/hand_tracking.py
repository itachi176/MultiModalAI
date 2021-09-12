import cv2 
import numpy as np
import time 
import os 
# import HandTrackingModule as hm 
import mediapipe as mp 


class handDetector():
    def __init__(self, mode = False, maxHands = 2, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detetionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, frame, draw=True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)
        return frame 

    def findPosition(self, frame, handNo=0, draw= True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h,w, c = frame.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                fingers = []
                lmList.append([id, cx, cy])
                
        return lmList
        

# mpHands = mp.solutions.hands
# hands = mpHands.Hands()
# mpDraw = mp.solutions.drawing_utils
ptime = 0
tipIds = [4,8,12,16,20]

def main():
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    ptime = 0
    while(True):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = detector.findHands(frame)
        lmList = detector.findPosition(frame)
        if len(lmList) != 0:
            # print(lmList[8])
            frame = cv2.circle(frame, (lmList[8][1], lmList[8][2]), 20, (255,0,0), cv2.FILLED)
            fingers = []
            if lmList[tipIds[0]][1] < lmList[tipIds[0]-2][1]:
                fingers.append(1)
            else:
                fingers.append(0)
                
            for id in range(1, 5):
                if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
                print(lmList[tipIds[id]][2],lmList[tipIds[id]-2][2])
                
            print(fingers)
        #display fps
        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime
        cv2.putText(frame, f'FPS: {int(fps)}',(400, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (255, 255, 0), 3)

        cv2.imshow("image", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()