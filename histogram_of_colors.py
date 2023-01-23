from sklearn.model_selection import train_test_split 

import matplotlib.pyplot as plt
import cv2
import numpy as np

train = np.loadtxt('./input/train.csv', delimiter=',', skiprows=1)
test = np.loadtxt('./input/test.csv', delimiter=',', skiprows=1)

# сохраняем разметку в отдельную переменную
train_label = train[:, 0]
# приводим размерность к удобному для обаботки виду
train_img = np.resize(train[:, 1:], (train.shape[0], 28, 28))
test_img = np.resize(test, (test.shape[0], 28, 28))



hist = cv2.calcHist(train_img,[0],None,[256],[0,256])
  
# plot the above computed histogram
plt.plot(hist, color='b')
plt.title('Image Histogram For Blue Channel GFG')
plt.show()