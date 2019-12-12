import numpy as np
import cv2
import imutils
from imutils import paths
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
def test_embeddings(i,detector,embedder):
    print(i)
    test_img = cv2.imread(i)
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
    if face.shape[0]*face.shape[1]*face.shape[2]==0:
        return 69, False
    faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
    embedder.setInput(faceBlob)
    vec = embedder.forward()
    done = True
    return vec, done
name_list = os.listdir("images")
pathd = {"protoPath": "models/deploy.prototxt","detectMod":"models/res10_300x300_ssd_iter_140000.caffemodel","embMod":"models/openface_nn4.small2.v1.t7"}
detector = cv2.dnn.readNetFromCaffe(pathd['protoPath'],pathd['detectMod'])
embedder = cv2.dnn.readNetFromTorch(pathd['embMod'])
knownEmbeddings = []
knownNames = []
for name in name_list:
    img_list = os.listdir("images/{}".format(name))
    for imageName in img_list:
        vec, done = test_embeddings("images/{}/".format(name)+imageName, detector, embedder)
        if done:
            knownNames.append(name)
            knownEmbeddings.append(vec.flatten())
data = {"embeddings": knownEmbeddings, "names": knownNames}
f = open("train_embeddings", "wb")
f.write(pickle.dumps(data))
f.close()


print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

# train the model used to accept the 128-d embeddings of the face and
# then produce the actual face recognition
print("[INFO] training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

# write the actual face recognition model to disk
f = open("recognizer", "wb")
f.write(pickle.dumps(recognizer))
f.close()

# write the label encoder to disk
f = open("labelEncoder", "wb")
f.write(pickle.dumps(le))
f.close()