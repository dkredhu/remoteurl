import numpy as np
import cv2

dataset = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_SIMPLEX

# Reshape
f_01 = np.load('face1.npy').reshape((20, 50*50*3))
f_02 = np.load('face2.npy').reshape((20, 50*50*3))
#f_03 = np.load('face3.npy').reshape((20, 50*50*3))

print(f_01.shape)
print(f_02.shape)
#print(f_03.shape)

names = {
    0 : 'RAKESH',
    1 : 'JAYANT',
    
}

labels = np.zeros((40,1))
labels[0:20, :] = 0.0    # first 20 for user_1
labels[20:40, :] = 1.0   # next 20 for user_2
#labels[40:, :] = 2.0     # last 20 for user_3

data = np.concatenate([f_01, f_02])
print(data.shape)

def distance(x1, x2):
    return np.sqrt(((x1 - x2) ** 2).sum())

def knn(x, train, target, k=5):
    m = train.shape[0]
    dist = []
    
    for i in range(m):
        dist.append(distance(x, train[i]))
        
    dist = np.asarray(dist)
    index = np.argsort(dist)
    
    sorted_labels = labels[index][:k]
    counts = np.unique(sorted_labels, return_counts = True)
    return counts[0][np.argmax(counts[1])]



while True:
	ret, img = cap.read()
	if ret:
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = dataset.detectMultiScale(gray, 1.3, 5)
		for (x, y, w, h) in faces:
			face_component = img[y:y+h, x:x+w, :]
			fc = cv2.resize(face_component, (50, 50))
			lab = knn(fc.flatten(), data, labels)
			text = names[int(lab)]
			cv2.putText(img, text, (x, y), font, 1, (255, 255, 0), 2)
			cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
		cv2.imshow('face recognition', img)
		if cv2.waitKey(33) == 27:
			break
	else:
		print('Error')
cap.release()
cv2.destroyAllWindows()





















