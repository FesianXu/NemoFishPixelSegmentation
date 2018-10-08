'''
@author: Fesian Xu
@date: 2018/9/20
@decription: Gray Image segment task.
@note: Originally using jupyter notebook as developing environment, in order to make this
program more universal, i transfer it into a standard python script file without any jupyter'scipy
magic command.
'''
import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import math

mask = sio.loadmat('./mats/Mask.mat')['Mask']
trainset = sio.loadmat('./mats/array_sample.mat')['array_sample']
testimg = cv2.imread('./mats/309.bmp')
testimg = np.array(testimg)
graytest = cv2.cvtColor(testimg, cv2.COLOR_BGR2GRAY) / 255
# read images and color space transform

label = trainset[:, -1]
gray_train = ((trainset[:, 0]))
rgb_train = ((trainset[:,1:4]))
# split the data

# this is for gray
# prior
pC0 = len(np.where(label == -1)[0])/len(label)
pC1 = len(np.where(label == 1)[0])/len(label)

# likelihood
c1_mean = np.sum(gray_train[np.where(label == -1)[0]])/len(gray_train[np.where(label == -1)[0]])
c1_sigma2 = np.sum(np.power(gray_train[np.where(label == -1)]-c1_mean,2))/len(gray_train[np.where(label == -1)[0]])

c2_mean = np.sum(gray_train[np.where(label == 1)[0]])/len(gray_train[np.where(label == 1)[0]])
c2_sigma2 = np.sum(np.power(gray_train[np.where(label == 1)]-c2_mean,2))/len(gray_train[np.where(label == 1)[0]])

x = np.where(mask == 1)[0]
y = np.where(mask == 1)[1]
predicts = np.zeros((0,3))

# visualize the segment result

for each in zip(x,y):
    x = graytest[each]
    poster_c1 = 1/np.sqrt(c1_sigma2) * np.exp(-(x-c1_mean)**2/(2*c1_sigma2)) * pC0
    poster_c2 = 1/np.sqrt(c2_sigma2) * np.exp(-(x-c2_mean)**2/(2*c2_sigma2)) * pC1
    predict = 0
    if poster_c1 > poster_c2:
        predict = -1
    else:
        predict = 1
    predict = np.array((each[0], each[1], predict))
    predict = predict[np.newaxis, :]
    predicts = np.concatenate((predicts,predict), axis=0)

newimg = np.zeros((240,320))
for each in predicts:
    if each[2] == 1:
        newimg[int(each[0]), int(each[1])] = 0.5
    else:
        newimg[int(each[0]), int(each[1])] = 1
plt.axis('off')
plt.imshow(newimg)
# show the image