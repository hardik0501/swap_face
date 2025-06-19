import cv2
from cvzone.FaceDetectionModule import FaceDetector
import numpy as np

cap = cv2.VideoCapture(0)
detector = FaceDetector(minDetectionCon=0.7)

print("Press '1' to capture first face")
print("Press '2' to capture second face")
print("Press 's' to swap faces")
print("Press 'q' to quit")

face1 = None
face2 = None

while True:
    success, img = cap.read()
    img, bboxs = detector.findFaces(img)

    key = cv2.waitKey(1)

    if key == ord('1') and bboxs:
        x, y, w, h = bboxs[0]['bbox']
        face1 = img[y:y+h, x:x+w].copy()
        print("Captured face 1")

    if key == ord('2') and bboxs:
        x, y, w, h = bboxs[0]['bbox']
        face2 = img[y:y+h, x:x+w].copy()
        print("Captured face 2")

    if key == ord('s') and face1 is not None and face2 is not None:
        img, bboxs = detector.findFaces(img)
        if len(bboxs) >= 2:
            for i, face in enumerate([face2, face1]):
                x, y, w, h = bboxs[i]['bbox']
                face_resized = cv2.resize(face, (w, h))
                img[y:y+h, x:x+w] = face_resized
            print("Faces swapped")

    if key == ord('q'):
        break

    cv2.imshow("Face Swap", img)

cap.release()
cv2.destroyAllWindows()
