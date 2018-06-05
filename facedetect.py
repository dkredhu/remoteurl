

import cv2
import numpy as np

dataset = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
print(dataset)

cap = cv2.VideoCapture(0)

data = []

while True:
    ret, img = cap.read()
    
    if ret:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = dataset.detectMultiScale(gray, 1.3)
        
        for x,y,w,h in faces:
            
            face_comp = img[y:y+h, x:x+w, :]
            fc = cv2.resize(face_comp, (50,50))
            
            if x % 10 == 0 and len(data) < 20:
                data.append(fc)
                print(data)
                print("Length of data is",len(data))
            
            cv2.rectangle(img, (x,y),(x+w,y+h),(0,0,255),5)
            
            cv2.imshow('img', img)
            
        if cv2.waitKey(1) == 27 or len(data) >= 20:
            break
    else:
        print('Error')

data = np.array(20)

print(data.shape)

np.save('face_2', data) 
         
cap.release()
cv2.destroyAllWindows()
