import cv2
import os

faceDetect = cv2.CascadeClassifier('assets/haarcascade/haarcascade_frontalface_default.xml')
capt = cv2.VideoCapture(0)
a = 1

while True:
    check, img = capt.read()
    faces = faceDetect.detectMultiScale(img, 1.3, 6)
    cv2.imshow('Faces', img)

    for(x, y, w, h) in faces:
        cv2.imwrite('test/test.jpg', img[y:y+h, x:x+w])
        
    if(len(faces) != 0): a = a + 1
    if a==10: break

    key = cv2.waitKey(1)

capt.release()
cv2.destroyAllWindows()