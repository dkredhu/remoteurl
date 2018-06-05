import cv2

dataset = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

image = cv2.imread('image_1.jpg', cv2.COLOR_BGR2GRAY)

faces = dataset.detectMultiScale(image, 1.3)

for x,y,w,h in faces:
    cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 5)
    
cv2.imwrite('result.jpg',image)