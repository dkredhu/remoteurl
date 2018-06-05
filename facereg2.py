#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 10:30:38 2018

@author: nishthadua
"""

import cv2
dataset = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
while True:
     ret, img = cap.read()
     if ret:
         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
         faces = dataset.detectMultiScale(gray, 1.3)
         for x,y,w,h in faces:
             cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 5)
         cv2.imshow('img',img)
         k = cv2.waitKey(30) & 0xFF
         if k == 27:
             break
     else:
        print("Error")
cap.release()
cv2.destroyAllWindow()



