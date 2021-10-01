import cv2
video1 = cv2.VideoCapture(0)
video2 = cv2.VideoCapture(1)

while True:
    ret0, frame0 = video1.read()
    ret1, frame1 = video2.read()

    if ret0:
        cv2.imshow("cam0", frame0)
        
    if ret1:
        cv2.imshow("cam1", frame1)

    if cv2.waitKey(1) == ord("q"):
        break

video1.release()
video2.release()
cv2.destroyAllWindows()