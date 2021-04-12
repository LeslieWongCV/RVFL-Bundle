# -*- coding: utf-8 -*-
# @Time    : 2021/4/11 5:18 下午
# @Author  : Yushuo Wang
# @FileName: main.py
# @Software: PyCharm
# @Blog    ：https://lesliewongcv.github.io/


import numpy as np
import os
from scipy.io import loadmat
from sklearn.model_selection import KFold


def sigmoid(a, b, x):

    return 1.0 / (1 + np.exp(-1.0 * (x.dot(a) + b)))


def dRVFL_prototype(X, T, C, n, L, n_node):
    '''
    Variables：X - input；No. of samples * No. of feature（N*n）
             ：H - H matrix；No. of samples * No. of hidden nodes（N*L）
             ：T - Target；No. of samples * No. of output nodes（N*M）
             ：C - Hyper-parm of the regularization
    '''
    # init the weight of hidden layers randomly
    a_list = []
    b_list = []
    T = one_hot(T)
    D = X.copy()
    H = X.copy()
    for i in range(layers):
        a = np.random.normal(0, 1, (H.shape[1], n_node))
        b = np.random.normal(0, 1)
        H = sigmoid(a, b, H)
        D = np.concatenate((D, H), axis=1)
        a_list.append(a)
        b_list.append(b)
        # calculate the weight of output layers(beta) and output
    DD = D.T.dot(D)
    DT = D.T.dot(T)
    if L > n:
        beta = np.linalg.pinv(DD + np.identity(DD.shape[0]) / C).dot(DT)
    else:
        beta = D.T.dot(np.linalg.pinv(D.dot(D.T) + np.identity(L + layers * n) / C)).dot(T)
    Fl = D.dot(beta)
    return beta, Fl, a_list, b_list


def one_hot(l):
    y = np.zeros([len(l), np.max(l)+1])
    for i in range(len(l)):
        y[i, l[i]] = 1
    return y


def predict(X, BETA, a, b):
    H = X.copy()
    for i in range(layers):
        H = sigmoid(a[i], b[i], H)
        X = np.concatenate((X, H), axis=1)
    Y = X.dot(BETA)
    Y = Y.argmax(1)
    return Y


def evaluation(y_hat, goundtruth):
    y_hat = y_hat[:, np.newaxis]
    return np.sum(np.equal(y_hat, goundtruth) / len(y_hat))


KFOLD = 4
PATH = '/Users/leslie/Downloads/MatDataset/'    # Path to the dataset
folders = os.listdir(PATH)
RES = []
C_list = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]  # hyper-pram 1/C
layers = 5
progress = 0

for folder_name in folders:
    progress += 100 / 29
    file_name = folder_name
    if folder_name == '.DS_Store':
        continue
    else:
        matfn = PATH + folder_name + '/' + folder_name + '_Train.mat'
        df_data = loadmat(matfn)['Data']
        df_label = loadmat(matfn)['Label']
        kf = KFold(n_splits=4, shuffle=False)

        for C in C_list:
            vali_res = 0

            for train_index, test_index in kf.split(df_label):  # 4-fold
                beta, Fl, A, B = dRVFL_prototype(df_data[train_index], df_label[train_index],
                                                 C=C, n=len(df_data[1]), L=len(train_index), n_node=100)
                y_valid = predict(df_data[test_index], beta, A, B)
                acc_valid = evaluation(y_valid, df_label[test_index])
                vali_res += acc_valid
            RES += [folder_name + ':' + str(vali_res/4) + '   C = ' + str(C)]
    print(str(round(progress)) + "%")  # show progress
RES = np.array(RES)
np.savetxt("dRVFL_acc_100_fix.txt", RES, fmt='%s', delimiter=',')