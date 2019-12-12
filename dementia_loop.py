import cv2
import numpy as np
from gtts import gTTS
import imutils
import pickle
import time
import pygame
import os
def cycle_names(recognizer, le, embedder,detections,w,h):
    name_list = []
    boxes = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            boxes.append(box)
            (startX, startY, endX, endY) = box.astype("int")
            face = frame[startY:endY, startX:endX] 
            if face.shape[0]*face.shape[1]*face.shape[2]!=0:
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
    				(96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()
                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                prob = preds[j]
                name = le.classes_[j]
                name_list.append(name)
    return name_list, boxes
def box_images(frame, boxes):
    for box in boxes:
        (startX, startY, endX, endY) = box.astype("int")
        cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
    return frame
pathd = {"protoPath": "models/deploy.prototxt","detectMod":"models/res10_300x300_ssd_iter_140000.caffemodel","embMod":"models/openface_nn4.small2.v1.t7",'recognizer':'recognizer','le':'labelEncoder'}
detector = cv2.dnn.readNetFromCaffe(pathd['protoPath'],pathd['detectMod'])
embedder = cv2.dnn.readNetFromTorch(pathd['embMod'])
recognizer = pickle.loads(open(pathd["recognizer"], "rb").read())
le = pickle.loads(open(pathd["le"], "rb").read())
camera = cv2.VideoCapture(0)
delay = 5
previous = time.time()
SONG_END = pygame.USEREVENT + 1
while (True):
    ret, frame = camera.read()
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]
    imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(frame, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)
    detector.setInput(imageBlob)
    detections = detector.forward()
    detector.setInput(imageBlob)
    detections = detector.forward()
    if detections.shape[2]==0:
        said_text = "There is no one present."
    else:
        name_list, boxes = cycle_names(recognizer,le,embedder,detections,w,h)
        frame = box_images(frame,boxes) 
        if len(name_list)==0:
            said_text = "There is no one present."
        elif len(name_list)==1:
            said_text = "{} is present".format(name_list[0])
        else:
            said_text = ""
            for c in range(len(name_list)):
                if c==len(name_list)-1:
                    said_text+=" and {} are present".format(name_list[c])
                else:
                    said_text+=" "+name_list[c]
    print(said_text)
    cv2.imshow('frame',frame)
    current_time = time.time()
    if current_time-previous>delay:
        print("Speaking...")
        pygame.mixer.init()
        previous = current_time
        tts = gTTS(said_text)
        tts.save("temp1.mp3")
        f = open('temp1.mp3')
        pygame.mixer.music.load(f)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy()==1:
            pygame.time.wait(100)
        f.close()
        pygame.mixer.quit()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()