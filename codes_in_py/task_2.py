'''
@author: Fesian Xu
@date: 2018/9/20
@decription: Color Image segment task.
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

# this is for rgb, but have the same prior as in gray one.
# prior
pC0 = len(np.where(label == -1)[0])/len(label)
pC1 = len(np.where(label == 1)[0])/len(label)

c1_means = np.sum(rgb_train[np.where(label == -1)[0]], axis=0)/len(rgb_train[np.where(label == -1)[0]])
c2_means = np.sum(rgb_train[np.where(label == 1)[0]], axis=0)/len(rgb_train[np.where(label == 1)[0]])

c1_sigma = np.dot(np.transpose(rgb_train[np.where(label == -1)[0]]-c1_means, (1,0)), rgb_train[np.where(label == -1)[0]]-c1_means)/len(rgb_train[np.where(label == -1)[0]])
c2_sigma = np.dot(np.transpose(rgb_train[np.where(label == 1)[0]]-c2_means, (1,0)), rgb_train[np.where(label == 1)[0]]-c2_means)/len(rgb_train[np.where(label == 1)[0]])

# gaussian
def cal(x, means, sigmas):
    inv = np.linalg.inv(sigmas)
    exp = -0.5*np.dot((x-means), inv)
    exp = np.dot(exp, (x-means))
    return np.exp(exp)/np.sqrt(np.linalg.det(sigmas))
	
# predict stage
x = np.where(mask == 1)[0]
y = np.where(mask == 1)[1]
predicts = np.zeros((0,3))
inn = testimg[:,:,::-1]/255
for each in zip(x,y):
    pix = inn[each]
    pix = np.array(pix)
    poster_c1 = cal(pix, c1_means, c1_sigma)*pC0
    poster_c2 = cal(pix, c2_means, c2_sigma)*pC1
    predict = 0
    if poster_c1 > poster_c2:
        predict = -1
    else:
        predict = 1
    predict = np.array((each[0], each[1], predict))
    predict = predict[np.newaxis, :]
    predicts = np.concatenate((predicts,predict), axis=0)

# visualize the segment result
newimg = np.zeros((240,320))
for each in predicts:
    if each[2] == 1:
        newimg[int(each[0]), int(each[1])] = 0.5
    else:
        newimg[int(each[0]), int(each[1])] = 1
		
plt.axis('off')
plt.imshow(newimg)
