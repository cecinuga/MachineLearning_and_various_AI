##                           PROGETTO FACERECOGNITION
#Creare un gioco dove la macchina conta le persone all'interno di un video animato di Xt 


import cv2 as cv
print(cv.__version__)
from random import randrange
print('Inizio Flusso.')

dati_persone = 'modelli/haarcascade_fullbody.xml' 
videosrc = 'modelli/genteperstrada.mp4'

video = cv.VideoCapture(videosrc) #STUDIA VideoCapture
tracker = cv.CascadeClassifier(dati_persone) #STUDIA CascadeClassifier
print('Modello importato.')


while True:
    (read_successful, frame) = video.read()
    if read_successful:
        grayscaled_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    else:
        break

    persone = tracker.detectMultiScale(grayscaled_frame)

    for (x, y, w, h) in persone:
        cv.rectangle(frame, (x, y), (x+h, y+h), (randrange(128, 255), randrange(128, 255), randrange(128, 255)), 2)

    cv.imshow('GrayScaled BodyDetector.', frame)
    key = cv.waitKey(1)

    if key==81 or key==113:
        break



#successfull_frame_read, frame = webcam.read()
#grayscaled_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#body_coordinates = trained_body_data.detectMultiScale(grayscaled_img) #STUDIA detectMultiScale
#
#for (x, y, w, h) in body_coordinates:
##cv.rectangle(grayscaled_img, (x, y), (x+w, y+h), (randrange(128,256), randrange(128,256), randrange(128,256),), 2) #STUDIA rectangle

