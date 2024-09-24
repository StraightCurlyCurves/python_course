import cv2 as cv

def nothing(x):
    pass

windowColorSelect = 'Select Color'

cv.namedWindow(windowColorSelect)

cv.createTrackbar("Hue low", windowColorSelect, 0, 179, nothing)
cv.createTrackbar("Hue high", windowColorSelect, 179, 179, nothing)
cv.createTrackbar("Sat low", windowColorSelect, 0, 255, nothing)
cv.createTrackbar("Sat high", windowColorSelect, 255, 255, nothing)
cv.createTrackbar("Val low", windowColorSelect, 0, 255, nothing)
cv.createTrackbar("Val high", windowColorSelect, 255, 255, nothing)

# create opencv camera object
cam = cv.VideoCapture(0)

while True:
    ret, img = cam.read()
    frame_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
    filtered = cv.inRange(frame_hsv,
        (
        cv.getTrackbarPos('Hue low', windowColorSelect),
        cv.getTrackbarPos('Sat low', windowColorSelect),
        cv.getTrackbarPos('Val low', windowColorSelect)),                                    
        (
        cv.getTrackbarPos('Hue high', windowColorSelect),
        cv.getTrackbarPos('Sat high', windowColorSelect),
        cv.getTrackbarPos('Val high', windowColorSelect)))

    cv.imshow("filtered", filtered)
    cv.imshow("orig", img)
    key = cv.waitKey(1)
    if key == ord('q'):
        break