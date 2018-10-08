'''
@author: Wenjun Liu
@date: 2018/9/21
@description: hard GMM model.
'''

import numpy as np
from scipy import io
import cv2
import math
import matplotlib.pyplot as plt
import random


class firt_work:
    def __init__(self, train_array_path="array_sample.mat", test_img="309.bmp", test_mask="Mask.mat"):
        # 读入训练数组，将其转化成np数组
        self.train_array = io.loadmat(train_array_path)['array_sample']

        # 读入测试图片
        self.test_img = cv2.imread(test_img)

        # 读入测试图片mask，转化成np数组
        self.test_mask = io.loadmat(test_mask)['Mask']

    def third_task(self):
        plt.figure()
        # draw_x = []
        # draw_y = []
        train_array = self.train_array
        self.arr = np.zeros(shape=self.test_img.shape)
        # mu = np.sum(train_array[:, 1:4], axis=0) / train_array.shape[0]
        # deta = np.dot(np.transpose(train_array[:, 1:4] - mu), (train_array[:, 1:4] - mu)) / train_array.shape[0]
        # num = 0
        # while num < 10:
            # 初始化
            # c = True
            # while c:
            #     c = False
            #     mu1 = np.random.normal(size=(3,), loc=mu, scale=0.1)
            #     mu2 = np.random.normal(size=(3,), loc= mu,scale=0.1)
            #     for i in mu1:
            #         if i < 0 or i > 1:
            #             c = True
            #     for j in mu2:
            #         if j < 0 or j > 1:
            #             c = True
            # print(mu1, mu2)
            # while True:
            #     deta1 = np.random.normal(size=(3, 3), loc=deta, scale=0.1)
            #     deta2 = np.random.normal(size=(3, 3), loc=deta, scale=0.1)
            #     if np.linalg.det(deta1) > 0 and np.linalg.det(deta2) > 0:
            #         break
            #
            # print(deta1,deta2)
            # while True:
            #     pi = random.gauss(0.5, 0.1)
            #     new_pi = random.gauss(0.5, 0.1)
            #     if pi != new_pi and 0 < pi < 1 and 0 < new_pi < 1:
            #         break
        pi = 0.6177546088463414
        new_pi = 0.4912798853159004
        mu1 = np.array([0.58120536, 0.54774505, 0.13260008])
        mu2 = np.array([0.6528733, 0.29901202, 0.31053961])
        deta1 = np.array([[0.01478135, -0.11980083, -0.17413719],
                          [0.01642694, 0.15692188, - 0.16267354],
                          [0.04869681, 0.06372776, - 0.02511735]])
        deta2 = np.array([[-0.02945353, 0.02131434, 0.1637306],
                          [0.0654345, 0.0752998, 0.27085998],
                          [0.07877265, 0.16632865, 0.04125937]])
        time = 0
        # goon = True
        print(pi, new_pi)
        while (new_pi != pi):
            # if time > 50 or pi == 1.0 or pi == 0.0:
            #     goon = False
            #     break

            # E
            pi = new_pi
            print("pi", pi, "u1", mu1, "u2", mu2)
            for i in range(train_array.shape[0]):
                a = math.exp(-0.5 * np.dot(np.dot((train_array[i, 1:4] - mu1), np.linalg.inv(deta1)),
                                           np.transpose((train_array[i, 1:4] - mu1)))) / math.sqrt(
                    np.linalg.det(deta1)) * pi
                b = math.exp(-0.5 * np.dot(np.dot((train_array[i, 1:4] - mu1), np.linalg.inv(deta2)),
                                           np.transpose((train_array[i, 1:4] - mu2)))) / math.sqrt(
                    np.linalg.det(deta2)) * (1 - pi)
                if a > b:
                    train_array[i, 4] = 1
                else:
                    train_array[i, 4] = 0
            # M
            train_array1 = train_array[np.where(train_array[:, 4] == 1)]
            new_pi = train_array1.shape[0] / train_array.shape[0]
            train_array2 = train_array[np.where(train_array[:, 4] == 0)]

            mu1 = np.sum(train_array1[:, 1:4], axis=0) / train_array1.shape[0]
            deta1 = np.dot(np.transpose(train_array1[:, 1:4] - mu1), (train_array1[:, 1:4] - mu1)) / \
                    train_array1.shape[0]
            mu2 = np.sum(train_array2[:, 1:4], axis=0) / train_array2.shape[0]
            deta2 = np.dot(np.transpose(train_array2[:, 1:4] - mu1), (train_array2[:, 1:4] - mu1)) / \
                    train_array2.shape[0]
            time += 1
            # if goon:
            #     draw_x.append(num)
            #     draw_y.append(time)
            #     print(draw_x, draw_y)
            #     num += 1
        # plt.plot(draw_x, draw_y)
        # plt.show()

        test_img = cv2.cvtColor(self.test_img, cv2.COLOR_BGR2RGB) / 255
        for i in range(test_img.shape[0]):
            for j in range(test_img.shape[1]):
                if self.test_mask[i, j] != 0:
                    a = math.exp(-0.5 * np.dot(np.dot((test_img[i, j] - mu1), np.linalg.inv(deta1)),
                                               np.transpose((test_img[i, j] - mu1)))) / math.sqrt(
                        np.linalg.det(deta1)) * \
                        train_array1.shape[0]

                    b = math.exp(-0.5 * np.dot(np.dot((test_img[i, j] - mu2), np.linalg.inv(deta2)),
                                               np.transpose((test_img[i, j] - mu2)))) / math.sqrt(
                        np.linalg.det(deta2)) * \
                        train_array2.shape[0]

                    if a > b:
                        self.arr[i, j] = np.array([0, 0, 255])

                    else:
                        self.arr[i, j] = np.array([0, 255, 255])
        cv2.imwrite("third_task.jpg", self.arr)
        cv2.imshow("3", self.arr)


f = firt_work()
f.first_task()

