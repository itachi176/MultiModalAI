import mtcnn 
import cv2
# img = cv2.imread("face.jpg")
cap = cv2.VideoCapture(0)
while(True):
    a = mtcnn.MTCNN()
    ret, frame = cap.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print(a.detect_faces(img)[0]['box'])
    box = a.detect_faces(img)[0]['box']
    img = cv2.rectangle(img, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (255, 0, 0), 2)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow('ah', img)
    # cv2.waitKey()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()