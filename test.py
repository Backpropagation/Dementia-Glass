# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:27:03 2019

@author: taych
"""
import numpy as np
import cv2
import imutils
from imutils import paths
pathd = {"protoPath": "models/deploy.prototxt","detectMod":"models/res10_300x300_ssd_iter_140000.caffemodel","embMod":"models/openface_nn4.small2.v1.t7"}
detector = cv2.dnn.readNetFromCaffe(pathd['protoPath'],pathd['detectMod'])
embedder = cv2.dnn.readNetFromTorch(pathd['embMod'])
def make_images():
    camera = cv2.VideoCapture(0)
    c = 0
    while (True):
        ret, frame = camera.read()
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.waitKey(1) & 0xFF == ord('b'):
            cv2.imwrite("images/frame_{}.jpg".format(c),frame)
            c+=1
    camera.release()
    cv2.destroyAllWindows()
    return
def test_embeddings(detector,embedder):
    test_img = cv2.imread("images/frame_0.jpg")
    image = imutils.resize(test_img, width=600)
    (h, w) = test_img.shape[:2]
    imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(image, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)
    #cv2.imshow('test',image)
    #cv2.waitKey(0)
    detector.setInput(imageBlob)
    predict = detector.forward()
    i = np.argmax(predict[0, 0, :, 2])
    confidence = predict[0, 0, i, 2]
    box = predict[0, 0, i, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    face = image[startY:endY, startX:endX]
    faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
    embedder.setInput(faceBlob)
    vec = embedder.forward()
    return
test_embeddings(detector,embedder)