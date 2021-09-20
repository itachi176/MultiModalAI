import cv2 
import numpy as np
import time 
import os 
# import HandTrackingModule as hm 
import mediapipe as mp 
import tensorflow as tf
# from gesture.test import pred
import numpy as np
model = tf.keras.models.load_model('./gesture/ml.h5')
def pred(img):
# img = cv2.imread('./test4.png')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(gray_img, 50, 255, cv2.THRESH_BINARY_INV)
    gray_img = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
    # img = cv2.resize(img, (28, 28))
    # img = np.pad(img, (10,10), 'constant', constant_values=0 )
    # img = cv2.resize(img, (28,28))/255
    # a = tf.expand_dims(img, axis=0)
    # x = model.predict(a)
    # y = np.argmax(x)
    # print(y)
    contours, hierarchy = cv2.findContours(gray_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Sắp xếp các contour theo diện tích giảm dần:
    area_cnt = [cv2.contourArea(cnt) for cnt in contours]
    area_sort = np.argsort(area_cnt)[::-1]
    # Top 20 contour có diện tích lớn nhất
    # print("area:", area_sort)
    def _drawBoundingBox(img, cnt):
        x,y,w,h = cv2.boundingRect(cnt)
        img = cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),2)
        new_img = img[y:y+h, x:x+w]
        return new_img
    new_img = _drawBoundingBox(img, contours[1])
    new_img= cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    new_img = cv2.resize(new_img, (28, 28))
    new_img = np.pad(new_img, (10,10), 'constant', constant_values=0 )
    new_img = cv2.resize(new_img, (28,28))/255
    # print(new_img.shape)
    a = tf.expand_dims(new_img, axis=0)
    x = model.predict(a)
    y = np.argmax(x)
    # print(y)
    return y

class handDetector():
    def __init__(self, mode = False, maxHands = 2, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detetionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4,8,12,16,20]

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
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h,w, c = frame.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                fingers = []
                self.lmList.append([id, cx, cy])
                
        return self.lmList
        
    def finger(self):
        fingers = []
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
            
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
            # print(self.lmList[self.tipIds[id]][2], self.lmList[self.tipIds[id]-2][2])
            
        return fingers

    
# mpHands = mp.solutions.hands
# hands = mpHands.Hands()
# mpDraw = mp.solutions.drawing_utils
ptime = 0

def main():
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    ptime = 0
    xp, yp = 0, 0
    # imgCanvas = np.ones((480, 640, 3), np.uint8)
    # model = tf.keras.models.load_model('./gesture/ml.h5')
    while(True):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = detector.findHands(frame)
        lmList = detector.findPosition(frame)
        # print(frame.shape)
        if len(lmList) != 0:
            # print(lmList[8])
            finger = detector.finger()
            
            

            # print(finger[2])
        
            # print(finger)
            # if finger[1] and finger[2] == False:
                # print("select mode")
                # imgCanvas = np.zeros((480, 640, 3), np.uint8)
                
            # if finger[1] and finger[2] and finger[3] and finger[4]:
                # frame = cv2.circle(frame, (lmList[8][1], lmList[8][2]), 20, (255,0,0), cv2.FILLED)
                # if xp ==0 and yp ==0:
                #     xp, yp = lmList[8][1], lmList[8][2]

                # cv2.line(frame, (xp, yp), (lmList[8][1], lmList[8][2]), (0, 0, 255), 20 )
                # cv2.line(imgCanvas, (xp, yp), (lmList[8][1], lmList[8][2]), (0, 0, 255), 20)
                # xp, yp = lmList[8][1], lmList[8][2]

                # img = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
                
        
        # imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        # _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        # imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        # frame = cv2.bitwise_and(frame, imgInv)
        # frame = cv2.bitwise_or(frame, imgCanvas)
        #display fps
        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime
        cv2.putText(frame, f'FPS: {int(fps)}',(400, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (255, 255, 0), 3)

        # frame = cv2.addWeighted(frame,0.5,imgCanvas,0.5,0)
        
        if cv2.waitKey(1) == ord('q'):

            # print("drawing")
            # print(img.shape)
            # cv2.imwrite("./test4.png", imgCanvas)
            break

        #predict
        # try:
        #     final = cv2.imread("test4.png")
        #     result = pred(final)
        #     # print("du doan:", result)
        #     cv2.putText(frame, f'Predict: {result}',(0, 70), cv2.FONT_HERSHEY_PLAIN,
        #             3, (255, 255, 0), 3)
        #         # break
        # except:
        #     pass
        
        
        cv2.imshow("image", frame)
        if cv2.waitKey(33) == ord("a"):
            if not finger[1] and not finger[2] and not finger[3] and not finger[4] and finger[0]:
                print("006")
            if finger[1] and finger[2] and finger[3] and finger[4] and finger[0]:
                print("003")
            # print(finger)
            if finger[1] and finger[2] and not finger[3] and not finger[4] and not finger[0]:
                print("005")
            elif finger[1] and not finger[2] and not finger[3] and not finger[4] and not finger[0]:
                print("004")
    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()