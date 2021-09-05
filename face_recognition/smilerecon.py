import cv2 as cv

face_detector = cv.CascadeClassifier('modelli/haarcascade_frontalface_default.xml')
smile_detector = cv.CascadeClassifier('modelli/haarcascade_smile.xml')
webcam = cv.VideoCapture(0)

while True:
    succesful_frame_read, frame = webcam.read()

    if not succesful_frame_read:
        break

    grayframe = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(grayframe)
    smiles = face_detector.detectMultiScale(grayframe, scaleFactor=2, minNeighbors=20)

    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (100, 200, 50), 2)

    for (x, y, w, h) in smiles:
        cv.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 200), 2)

    cv.imshow('Smile Please.', frame)
    key = cv.waitKey(1)
    if key == 81 or key == 113:
        break

webcam.release()
cv.destroyAllWindows()