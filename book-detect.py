import cv2
import os

capt = cv2.VideoCapture(0)
img_counter = 0

while True:
    check, img = capt.read()

    cv2.imshow('Faces', img)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "assets/books/test.jpg".format(img_counter)
        cv2.imwrite(img_name, img)
        print('Capturing image book successfully!')
        break

    key = cv2.waitKey(1)

capt.release()
cv2.destroyAllWindows()