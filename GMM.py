import numpy as np
import cv2 as cv
import os
from numpy.linalg import norm, inv

init_weight = [0.7, 0.1, 0.1, 0.1]
init_u = np.zeros(3)
# initial Covariance matrix
init_sigma = 225*np.eye(3)
init_alpha = 0.01
# prevent deviding 0 for stability
epsilon = 0.00000001


class GMM():
    def __init__(self, data_dir, train_num, alpha=init_alpha):
        self.data_dir = data_dir
        self.train_num = train_num
        self.alpha = alpha
        self.img_shape = None

        self.weight = None
        self.mu = None
        self.sigma = None
        self.K = None
        self.B = None
        self.weight_order = None

    def check(self, pixel, mu, sigma):
        '''
        check whether a pixel match a Gaussian distribution. Matching means pixel is less than
        2.5 standard deviations away from a Gaussian distribution.
        '''
        x = np.mat(np.reshape(pixel, (3, 1)))
        u = np.mat(mu).T
        sigma = np.mat(sigma)
        # calculate Mahalanobis distance
        d = np.sqrt((x-u).T*sigma.I*(x-u))
        if d < 2.5:
            return True
        else:
            return False

    def train(self, K=4):
        '''
        train model
        '''
        self.K = K
        file_list = []
        # file numbers are from 1 to train_number
        for i in range(self.train_num):
            file_name = os.path.join(self.data_dir, 'b%05d' % i + '.bmp')
            file_list.append(file_name)

        img_init = cv.imread(file_list[0])
        img_shape = img_init.shape
        self.img_shape = img_shape
        self.weight = np.array([[init_weight for j in range(self.img_shape[1])] for i in range(self.img_shape[0])])
        self.mu = np.array([[[init_u for k in range(self.K)] for j in range(img_shape[1])]
                             for i in range(img_shape[0])])
        self.sigma = np.array([[[init_sigma for k in range(self.K)] for j in range(img_shape[1])]
                             for i in range(img_shape[0])])
        self.B = np.ones(self.img_shape[0:2], dtype=np.int)
        self.weight_order = np.zeros(self.mu.shape[0:-1], dtype=np.int)
        for i in range(img_shape[0]):
            for j in range(img_shape[1]):
                for k in range(self.K):
                    self.mu[i][j][k] = np.array(img_init[i][j]).reshape(1,3)
        for i in range(self.K):
            print('u:{}'.format(self.mu[100][100][i]))
        # update process
        for file in file_list:
            print('training:{}'.format(file))
            img=cv.imread(file)
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    # Check whether match the existing K Gaussian distributions
                    flag = 0
                    for k in range(K):
                        if self.check(img[i][j], self.mu[i][j][k], self.sigma[i][j][k]):
                            flag = 1
                            M = 1
                            self.weight[i][j][k] = self.weight[i][j][k] + self.alpha*(M - self.weight[i][j][k])
                            u = self.mu[i][j][k]
                            sigma = self.sigma[i][j][k]
                            x = img[i][j].astype(np.float)
                            delta = x - u
                            self.mu[i][j][k] = u + M*(self.alpha / (self.weight[i][j][k]+epsilon))*delta
                            self.sigma[i][j][k] = sigma + M*(self.alpha / (self.weight[i][j][k]+epsilon))\
                                                            *(np.matmul(delta, delta.T)-sigma)
                        else:
                            m=0
                            self.weight[i][j][k] = self.weight[i][j][k] + self.alpha*(m-self.weight[i][j][k])
                    # if none of the K distributions match the current value
                    # the least probable distribution is replaced with a distribution
                    # with current value as its mean, an initially high variance and low rior weight
                    if flag == 0:
                        w_list = [self.weight[i][j][k] for k in range(K)]
                        id = w_list.index(min(w_list))
                        # weight keep same, replace mean with current value and set high variance
                        self.mu[i][j][id] = np.array(img[i][j]).reshape(1,3)
                        self.sigma[i][j][id] = np.array(init_sigma)
                    # normalize the weight
                    s = sum([self.weight[i][j][k] for k in range(K)])
                    for k in range(K):
                        self.weight[i][j][k] /= s
            print('img:{}'.format(img[100][100]))
            print('weight:{}'.format(self.weight[100][100]))
            self.reorder()
            for i in range(self.K):
                print('u:{}'.format(self.mu[100][100][i]))


    def reorder(self, T=0.75):
        '''
        reorder the estimated components based on the ratio pi / the norm of standard deviation.
        the first B components are chosen as background components
        the default threshold is 0.75
        '''
        for i in range(self.img_shape[0]):
            for j in range(self.img_shape[1]):
                k_weight = self.weight[i][j]
                k_norm = np.array([norm(np.sqrt(inv(self.sigma[i][j][k]))) for k in range(self.K)])
                ratio = k_weight/k_norm
                descending_order = np.argsort(-ratio)
                self.weight_order[i][j] = descending_order
                cum_weight = 0
                for index, order in enumerate(descending_order):
                    cum_weight += self.weight[i][j][order]
                    if cum_weight > T:
                        self.B[i][j] = index + 1
                        break


    def infer(self, img):
        '''
        infer whether its background or foregound
        if the pixel is background, both values of rgb will set to 255. Otherwise not change the value
        '''
        result = np.array(img)

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                for k in range(self.B[i][j]):
                    if self.check(img[i][j], self.mu[i][j][k], self.sigma[i][j][k]):
                        # [255, 255, 255] is white, the background color will be set to white
                        result[i][j] = [255, 255, 255]
                        break
                # gaussian_pixel = self.g_mat.mat[i][j]
                # for g in range(self.K):
                #     if self.check(img[i][j], gaussian_pixel[g]) and self.g_mat.weight[i][j][g] > 0.25:
                #         # [255, 255, 255] is white, the background color will be set to white
                #         result[i][j] = [255, 255, 255]
                #         continue
        return result