import numpy as np
import cv2 as cv
import os
from numba import jit, njit
from numpy.linalg import norm, inv

# initial Covariance matrix
init_sigma = 225*np.eye(3)
init_u = None
init_alpha = 0.01
# prevent deviding 0 for stability
epsilon = 0.00000001

class Gaussian():
    def __init__(self, u, sigma):
        self.u = u
        self.sigma = sigma


class GaussianMat():
    def __init__(self, shape, k):
        self.shape = shape
        self.k = k
        # initialize Gaussian distribution
        g = np.array([Gaussian(init_u, init_sigma) for i in range(k)])
        self.mat = np.array([[[Gaussian(init_u, init_sigma) for i in range(k)] for j in range(shape[1])]
                             for l in range(shape[0])])
        # intialize weight, it could be [1,0,0,0]，but we choose [0.7,0.1,0.1,0.1] for stability
        self.weight = np.array([[[0.7, 0.1, 0.1, 0.1] for j in range(shape[1])] for l in range(shape[0])])


class GMM():
    def __init__(self, data_dir, train_num, alpha=init_alpha):
        self.data_dir = data_dir
        self.train_num = train_num
        self.alpha = alpha
        self.g_mat = None
        self.K = None
        self.B = None
        self.weight_order = None

    def check(self, pixel, gaussian):
        '''
        check whether a pixel match a Gaussian distribution. Matching means pixel is less than
        2.5 standard deviations away from a Gaussian distribution.
        '''
        u = np.mat(gaussian.u).T
        x = np.mat(np.reshape(pixel,(3,1)))
        sigma = np.mat(gaussian.sigma)
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
        # file numbers are from 0 to 199
        for i in range(self.train_num):
            file_name = os.path.join(self.data_dir, 'b%05d'%i + '.bmp')
            file_list.append(file_name)
        img_init = cv.imread(file_list[0])
        img_shape = img_init.shape
        self.g_mat = GaussianMat(img_shape, K)
        self.B = np.ones(img_shape[0:2], dtype=np.int)
        self.weight_order = np.zeros(self.g_mat.mat.shape, dtype=np.int)

        for i in range(img_shape[0]):
            for j in range(img_shape[1]):
                for k in range(self.K):
                    self.g_mat.mat[i][j][k].u = np.array(img_init[i][j]).reshape(1,3)

        for i in range(self.K):
            print('u:{}'.format(self.g_mat.mat[100][100][i].u))

        # update process
        for file in file_list:
            print('training:{}'.format(file))
            img=cv.imread(file)
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    # Check whether match the existing K Gaussian distributions
                    flag = 0
                    for k in range(K):
                        if self.check(img[i][j], self.g_mat.mat[i][j][k]):
                            flag = 1
                            M = 1
                            self.g_mat.weight[i][j][k] = self.g_mat.weight[i][j][k] + \
                                                         self.alpha * (M - self.g_mat.weight[i][j][k])
                            u = self.g_mat.mat[i][j][k].u
                            sigma = self.g_mat.mat[i][j][k].sigma
                            x = img[i][j].astype(np.float)
                            delta = x - u
                            self.g_mat.mat[i][j][k].u = u + M*(self.alpha/(self.g_mat.weight[i][j][k]+epsilon))*delta
                            self.g_mat.mat[i][j][k].sigma = sigma + M*(self.alpha/(self.g_mat.weight[i][j][k]+epsilon))\
                                                            *(np.matmul(delta, delta.T)-sigma)
                        else:
                            m=0
                            self.g_mat.weight[i][j][k] = self.g_mat.weight[i][j][k] + self.alpha*(m-self.g_mat.weight[i][j][k])
                    # if none of the K distributions match the current value
                    # the least probable distribution is replaced with a distribution
                    # with current value as its mean, an initially high variance and low rior weight
                    if flag == 0:
                        w_list = [self.g_mat.weight[i][j][k] for k in range(K)]
                        id = w_list.index(min(w_list))
                        # weight keep same, replace mean with current value and set high variance
                        self.g_mat.mat[i][j][id].u = np.array(img[i][j]).reshape(1,3)
                        self.g_mat.mat[i][j][id].sigma = np.array(init_sigma)
                    # normalize the weight
                    s = sum([self.g_mat.weight[i][j][k] for k in range(K)])
                    for k in range(K):
                        self.g_mat.weight[i][j][k] /= s
            print('img:{}'.format(img[100][100]))
            print('weight:{}'.format(self.g_mat.weight[100][100]))
            self.reorder()
            for i in range(self.K):
                print('u:{}'.format(self.g_mat.mat[100][100][i].u))

    def reorder(self, T=0.75):
        '''
        reorder the estimated components based on the ratio pi / the norm of standard deviation.
        the first B components are chosen as background components
        the default threshold is 0.75
        '''

        for i in range(self.g_mat.shape[0]):
            for j in range(self.g_mat.shape[1]):
                k_weight = self.g_mat.weight[i][j]
                k_norm = np.array([norm(np.sqrt(inv(self.g_mat.mat[i][j][k].sigma))) for k in range(self.K)])
                ratio = k_weight/k_norm
                descending_order = np.argsort(-ratio)
                self.weight_order[i][j] = descending_order
                cum_weight = 0
                for index, order in enumerate(descending_order):
                    cum_weight += self.g_mat.weight[i][j][order]
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
                gaussian_mat = self.g_mat.mat[i][j]
                for k in range(self.B[i][j]):
                    if self.check(img[i][j], gaussian_mat[k]):
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
