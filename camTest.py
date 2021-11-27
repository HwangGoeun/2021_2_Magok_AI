import cv2

cap = cv2.VideoCapture(0)

if cap.isOpened() == False:
    print("Can't use cam")
    exit(1)

while(True):
    ret, img_frame = cap.read()

    if ret == False:
        print("Capture False")
        break

    cv2.imshow('Color', img_frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()