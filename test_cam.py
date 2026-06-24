import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
print("Opened:", cap.isOpened(), "| Got frame:", ret, "| Frame mean brightness:", frame.mean() if ret else None)
cv2.imshow("Test", frame)
cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()