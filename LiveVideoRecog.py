from keras.models import load_model
model=load_model("face_vgg.h5")
import numpy as np

import os
import IPython.display as ipd
import webbrowser as wb
import cv2
from keras.preprocessing import image
import PIL
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def face_detector(img ,size=0.5):
    faces = face_classifier.detectMultiScale(img, 1.3 ,5)
    if faces is ():
        return img, []
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0 ,255, 255),2)
        roi = img[y:y+h,x:x+w]
        roi = cv2.resize(roi, (224, 224))
    return img, roi

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    image, face = face_detector(frame)
    try:
        face=np.array(face)
        face=np.expand_dims(face, axis=0)
        if(face.shape==(1,0)):
            cv2.putText(image, "Not Deteted", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv2.imshow('Face Recognition',image)
            print("3")
            pass
        else:
            results=model.predict(face)
            if(results[0][0] == 0.0):
                cv2.putText(image, "Rohit", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                cv2.imshow('Face Recognition',image)
                print("1")
            elif(results[0][0] != 0.0):
                cv2.putText(image, "Rahul", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                cv2.imshow('Face Recognition',image)
                print("2")
                
    except:
        cv2.imshow('Face Recognition', image )
        cv2.putText(image, "No Face Found", (220, 120) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        print("4")
        pass
    if  cv2.waitKey(1) == 13: #13 is the Enter Key
        break
cap.release()
cv2.destroyAllWindows()
cap.release()