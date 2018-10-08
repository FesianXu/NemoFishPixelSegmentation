'''
@author: Fesian Xu
@date: 2018/9/20
@decription: Unsupervised learning using GMM model based on EM algorithm to approximate the parameters.(soft pi_j)
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

label = trainset[:, -1]
gray_train = ((trainset[:, 0]*255) // 1).astype(np.uint8)
rgb_train = ((trainset[:,1:4]*255) // 1).astype(np.uint8)


# soft
def gamma(x, pis, means, sigmas):
    def cal(x, mean, sigma):
        invsigma = np.linalg.inv(sigma)
        exp = -0.5*np.dot((x-mean),invsigma)
        exp = np.exp(np.dot(exp, x-mean))/np.sqrt(np.linalg.det(sigma))
        return exp
    softmaxs = []
    for each in zip(pis,means,sigmas):
        mean = each[1]
        pi = each[0]
        sigma = each[2]
        softmaxs.append(cal(x, mean, sigma) * pi)
    softmaxs = np.array(softmaxs)
    softmaxs = softmaxs/np.sum(softmaxs)
#     softmaxs = np.exp(softmaxs)/np.sum(np.exp(softmaxs))
   
    return softmaxs

def predict(x,means,sigmas):
    def cal(x, mean, sigma):
        invsigma = np.linalg.inv(sigma)
        exp = -0.5*np.dot((x-mean),invsigma)
        exp = np.exp(np.dot(exp, x-mean))/np.sqrt(np.linalg.det(sigma))
        return exp
    softmaxs = []
    for each in zip(means, sigmas):
        mean = each[0]
        sigma = each[1]
        softmaxs.append(cal(x,mean,sigma))
    softmaxs = np.array(softmaxs)
    return np.argmax(softmaxs)

# compute the time costing
import time
img_input = rgb_train
# initiate
c1_means = np.random.uniform(size=(3,))*255
c2_means = np.random.uniform(size=(3,))*255

while True:
    c1_sigmas = np.random.uniform(size=(3,3))*255
    c2_sigmas = np.random.uniform(size=(3,3))*255
    if np.linalg.det(c1_sigmas) > 0 and np.linalg.det(c2_sigmas) > 0:
        break

pi0 = 0.60
pi1 = 0.40

begin = time.clock()
# here we go!
while True:
    gammas = np.zeros((0,2))
    # Expection step here
    for pix in img_input:
        gamma_nk = gamma(pix, (pi0, pi1), means=(c1_means, c2_means),sigmas=(c1_sigmas, c2_sigmas))
        gamma_nk = np.array(gamma_nk)
        gamma_nk = gamma_nk[np.newaxis, :]
        gammas = np.concatenate((gammas, gamma_nk), axis=0)

    n0 = np.sum(gammas[:,0])
    n1 = np.sum(gammas[:,1])

    c1_means = np.sum(gammas[:,0][:,np.newaxis]*img_input, axis=0)/n0
    c2_means = np.sum(gammas[:,1][:,np.newaxis]*img_input, axis=0)/n1

    array_pixs = img_input
    # Maximum step here
    c1 = np.dot(np.transpose((array_pixs-c1_means), (1,0)), (array_pixs-c1_means))
    c1_sigmas = np.sum(np.array(list(map(lambda x:x*c1, gammas[:,0]))),axis=0)/n0

    c2 = np.dot(np.transpose((array_pixs-c2_means), (1,0)), (array_pixs-c2_means))
    c2_sigmas = np.sum(np.array(list(map(lambda x:x*c2, gammas[:,1]))),axis=0)/n1

    last_pi0 = pi0
    last_pi1 = pi1
    print(pi0)
    print(pi1)
    pi0 = n0/len(gammas[:,0])
    pi1 = n1/len(gammas[:,1])

    if abs(last_pi0-pi0) < 1e-5 and abs(last_pi1-pi1) < 1e-5:
        break
end = time.clock()
print(end-begin)


# visualize the segment result
x = np.where(mask == 1)[0]
y = np.where(mask == 1)[1]
predicts = np.zeros((0,3))
inn = testimg[:,:,::-1]
for each in zip(x,y):
    pix = inn[each]
    pred = predict(pix, means=(c1_means, c2_means), sigmas=(c1_sigmas, c2_sigmas))
    pred = np.array((each[0], each[1], pred))
    pred = pred[np.newaxis, :]
    predicts = np.concatenate((predicts,pred), axis=0)

newimg = np.zeros((240,320))
for each in predicts:
    if each[2] == 1:
        newimg[int(each[0]), int(each[1])] = 0.5
    else:
        newimg[int(each[0]), int(each[1])] = 1
plt.imshow(newimg)

