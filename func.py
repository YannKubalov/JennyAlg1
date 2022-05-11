import math

import numpy as np
from PIL import Image


def matrx_to_vec(matrx):
    vector = []
    for i in range(matrx.shape[0]):
        for j in range(matrx.shape[1]):
            vector.append(matrx.at[i, j])
    return vector


def vector_to_matrx(vector, size):
    matrx = np.zeros((int(len(vector) / size), size))
    for i in range(matrx.shape[0]):
        for j in range(matrx.shape[1]):
            matrx[i, j] = vector[i * size + j]
    return matrx


def matrx_to_PNG(data, file="hopfield.png"):
    y = np.zeros(data.shape, dtype=np.uint8)
    y[data == 1] = 255
    y[data == -1] = 0
    print(y)
    img = Image.fromarray(y, mode="L")
    img.save(file)
    return img


def get_hopfield(prev_ax, weights):
    new_ax = []
    for j in range(len(prev_ax)):
        val = 0
        for i in range(len(prev_ax)):
            val += weights[i, j] * prev_ax[i]
        if val < 0:
            new_ax.append(-1)
        else:
            new_ax.append(1)
    return new_ax


def get_hamming(prev_ax, weights, t=0):
    new_ax = []
    for j in range(weights.shape[1]):
        val = 0
        for i in range(len(prev_ax)):
            val += weights[i, j] * prev_ax[i]
        val += t
        new_ax.append(val)
    return new_ax


def get_new_hamming(prev_ax, eps=0.1):
    new_ax = []
    for j in range(4):
        val = prev_ax[j]
        deduct = 0
        for i in range(len(prev_ax)):
            if i != j:
                deduct += prev_ax[i]
        val = val - eps * deduct
        if val >= 0:
            new_ax.append(val)
        else:
            new_ax.append(0)
    return new_ax


def has_change(prev_ax, new_ax):
    boolean = False
    for i in range(len(prev_ax)):
        if prev_ax[i] != new_ax[i]:
            boolean = True
    return boolean


def compare_two_vector(prev_vector, new_vector, eps):
    dif = 0
    for i in range(len(prev_vector)):
        dif += math.pow(prev_vector[i] - new_vector[i], 2)
    if math.pow(dif, 1 / 2) < eps:
        return False
    else:
        return True


def start_hopfield(train, test, cols, rows):
    vec_train = matrx_to_vec(train)
    stands = int(len(vec_train)/len(matrx_to_vec(test)))
    weights = np.zeros((cols * rows, cols * rows))
    for i in range(weights.shape[0]):
        for j in range(i, weights.shape[1]):
            if i != j:
                for z in range(0, stands):
                    weights[i][j] += vec_train[i + z * cols * rows] * vec_train[j + z * cols * rows]
                weights[j][i] = weights[i][j]
    test = matrx_to_vec(test)
    prev = test
    infinity = True
    count = 0
    while infinity:
        new = get_hopfield(prev, weights)
        infinity = has_change(prev, new)
        prev = new
        count += 1
        if count % 1000 == 0:
            print(f"Количество итераций: {count}")
        if count > 100000:
            break
    prev = vector_to_matrx(prev, cols)
    matrx_to_PNG(prev)


def start_hamming(train, test):
    cols = train.shape[1]
    rows = int(train.shape[0] / 4)
    weights = np.zeros((cols * rows, 4))
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            weights[i][j] = train.at[(i // cols) + j * rows, i % cols] / 2
    test = matrx_to_vec(test)
    prev = get_hamming(test, weights, t=cols * rows / 2)
    infinity = True
    count = 0
    while infinity:
        new = get_new_hamming(prev, eps=0.01)
        infinity = compare_two_vector(prev, new, eps=0.1)
        prev = new
        count += 1
        if count % 1000 == 0:
            print(f"Количество итераций: {count}")
        if count > 10000:
            break
    print(prev)