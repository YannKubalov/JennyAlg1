import math

import pandas as pd
import numpy as np
from PIL import Image


def matx_to_vec(frame):
    vec = []
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            vec.append(frame.at[i, j])
    return vec


def vec_to_mat(vec, shape):
    matx = np.zeros((int(len(vec)/shape), shape))
    for i in range(matx.shape[0]):
        for j in range(matx.shape[1]):
            matx[i, j] = vec[i*shape+j]
    return matx


def array2img(data, file="neu.png"):
    y = np.zeros(data.shape, dtype=np.uint8)
    y[data == 1] = 255
    y[data == -1] = 0
    print(y)
    img = Image.fromarray(y, mode="L")
    img.save(file)
    return img


def cal_of_neu_hopfield(axons_prev, weights):
    axons_new = []
    for j in range(len(axons_prev)):
        val = 0
        for i in range(len(axons_prev)):
            val += weights[i, j] * axons_prev[i]
        if val < 0:
            axons_new.append(-1)
        else:
            axons_new.append(1)
    return axons_new


def cal_of_neu_hamming(axons_prev, weights, t=0):
    axons_new = []
    for j in range(weights.shape[1]):
        val = 0
        for i in range(len(axons_prev)):
            val += weights[i, j] * axons_prev[i]
        val += t
        axons_new.append(val)
    return axons_new


def cal_of_ax_hamming(axons_prev, eps=0.1):
    axons_new = []
    for j in range(4):
        val = axons_prev[j]
        deduct = 0
        for i in range(len(axons_prev)):
            if i != j:
                deduct += axons_prev[i]
        val = val - eps * deduct
        if val >= 0:
            axons_new.append(val)
        else:
            axons_new.append(0)
    return axons_new


def has_change(axons_prev, axons_new):
    for i in range(len(axons_prev)):
        if axons_prev[i] != axons_new[i]:
            return True
    return False


def compare_two_vec(pr_vec, new_vec, eps):
    dif = 0
    for i in range(len(pr_vec)):
        dif += math.pow(pr_vec[i] - new_vec[i], 2)
    if math.pow(dif, 1/2) < eps:
        return False
    else:
        return True


def init_hopfield(train, test):
    n_col = train.shape[1]
    n_row = int(train.shape[0] / 4)
    weights = np.zeros((n_col * n_row, n_col * n_row))
    for i in range(weights.shape[0]):
        for j in range(i, weights.shape[1]):
            if i != j:
                weights[i][j] = train.at[i // n_col, i % n_col] * train.at[j // n_col, j % n_col] + \
                                train.at[(i // n_col) + n_col, i % n_col] * train.at[(j // n_col) + n_col, j % n_col] +\
                                train.at[(i // n_col) + n_col * 2, i % n_col] * train.at[
                                    (j // n_col) + n_col * 2, j % n_col] + \
                                train.at[(i // n_col) + n_col * 3, i % n_col] * train.at[
                                    (j // n_col) + n_col * 3, j % n_col]
                weights[j][i] = weights[i][j]
    test = matx_to_vec(test)
    pr_ax = test
    indicator = True
    count = 0
    while indicator:
        n_ax = cal_of_neu_hopfield(pr_ax, weights)
        indicator = has_change(pr_ax, n_ax)
        pr_ax = n_ax
        count += 1
        if count % 1000 == 0:
            print(f"Количество итераций: {count}")
        if count > 100000:
            break
    pr_ax = vec_to_mat(pr_ax, n_col)
    array2img(pr_ax)


def init_hamming(train, test):
    n_col = train.shape[1]
    n_row = int(train.shape[0] / 4)
    weights = np.zeros((n_col * n_row, 4))
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            weights[i][j] = train.at[(i // n_col) + j*n_row, i % n_col] / 2
    test = matx_to_vec(test)
    pr_ax = cal_of_neu_hamming(test, weights, t=1/n_col*n_row)
    indicator = True
    count = 0
    while indicator:
        n_ax = cal_of_ax_hamming(pr_ax, eps=0.01)
        indicator = compare_two_vec(pr_ax, n_ax, eps=0.1)
        pr_ax = n_ax
        count += 1
        if count % 1000 == 0:
            print(f"Количество итераций: {count}")
        if count > 10000:
            break
    print(pr_ax)


train_sample = pd.read_excel('examples.xlsx', usecols="B:I", names=[0, 1, 2, 3, 4, 5, 6, 7], nrows=32, engine='openpyxl')
test_sample = pd.read_excel('examples.xlsx', usecols="B:I", names=[0, 1, 2, 3, 4, 5, 6, 7], skiprows=32, engine='openpyxl')

init_hopfield(train_sample, test_sample)
# init_hamming(train_sample, test_sample)
