import cv2
import os

a = 1

faceDetect = cv2.CascadeClassifier('assets/haarcascade/haarcascade_frontalface_default.xml')
capt = cv2.VideoCapture(0)

id = input("Masukkan NIS : ")
os.mkdir('assets/datasets/'+id)

while True:
    check, img = capt.read()
    faces = faceDetect.detectMultiScale(img, 1.3, 6)
    cv2.imshow('Faces', img)

    for(x, y, w, h) in faces:
        cv2.imwrite('assets/datasets/'+str(id)+'/user.'+str(id)+'.'+str(a)+'.jpg', img[y:y+h, x:x+w])
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
        
    if(len(faces) != 0): a = a + 1

    key = cv2.waitKey(1)
    if(a >= 251):
        break

    print("Photo "+str(a))

capt.release()
cv2.destroyAllWindows()