from dataclasses import dataclass
import cv2
import dib
from imutils import face_utils

cap = cv2.VideoCapture('Shia.mp4')


PREDICTOR_PATH ='68.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    face_rects = detector(frame,1)
    for (i, rect) in enumerate(face_rects):
        shape = predictor(frame, rect)
        shape = face_utils.shape_to_np(shape)

        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    cv2.imshow("Detection", frame)
    cv2.waitKey(0)
 
cap.release()
cv2.destroyAllWindows()