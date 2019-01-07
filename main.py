import GMM
import glob
import cv2 as cv
import os
import numpy as np

if __name__ == '__main__':
    data_dir = r'./WavingTrees'
    train_num = 200
    gmm = GMM.GMM(data_dir=data_dir, train_num=train_num)
    gmm.train()
    print('train finished')
    file_list = glob.glob(r'./WavingTrees/b*.bmp')
    file_index = 0
    for file in file_list:
        print('infering:{}'.format(file))
        img = cv.imread(file)
        img = gmm.infer(img)
        cv.imwrite(r'./output/'+'%05d'%file_index+'.bmp', img)
        file_index += 1