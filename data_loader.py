"""
Data generator for ModelNet40
reference: https://github.com/garyli1019/pointnet-keras
Date: 08/13/2018
Author: Tianzhong
"""
import numpy as np
import h5py
import random
from keras.utils import np_utils
from keras.utils import Sequence
import random

def ypr_rand(min_deg, max_deg):
    y = [random.uniform(min_deg, max_deg)]
    p = [random.uniform(min_deg, max_deg)]
    r = [random.uniform(min_deg, max_deg)]
    return y, p, r

def d_rand(min_d, max_d):
    d = [random.uniform(min_d, max_d), random.uniform(min_d, max_d), random.uniform(min_d, max_d)]
    return d

def q_rand():
    pn = 0
    while pn < 1e-5:
        p = [random.random(), 2 * (random.random() - 0.5), 2 * (random.random() - 0.5), 2 * (random.random() - 0.5)]
        pn = np.linalg.norm(p)
    return p / pn

def e2q(yaw, pitch, roll):

    yaw_rad = (np.deg2rad(yaw) / 2)
    pitch_rad = (np.deg2rad(pitch) / 2)
    roll_rad = (np.deg2rad(roll) / 2)
    qo = np.cos(roll_rad / 2) * np.cos(pitch_rad / 2) * np.cos(yaw_rad / 2) + np.sin(roll_rad / 2) * np.sin(pitch_rad / 2) * np.sin(yaw_rad / 2)
    qx = np.sin(roll_rad / 2) * np.cos(pitch_rad / 2) * np.cos(yaw_rad / 2) - np.cos(roll_rad / 2) * np.sin(pitch_rad / 2) * np.sin(yaw_rad / 2)
    qy = np.cos(roll_rad / 2) * np.sin(pitch_rad / 2) * np.cos(yaw_rad / 2) + np.sin(roll_rad / 2) * np.cos(pitch_rad / 2) * np.sin(yaw_rad / 2)
    qz = np.cos(roll_rad / 2) * np.cos(pitch_rad / 2) * np.sin(yaw_rad / 2) - np.sin(roll_rad / 2) * np.sin(pitch_rad / 2) * np.cos(yaw_rad / 2)
    return np.squeeze([qo, qx, qy, qz])

def qnorm(p):
    pn = np.linalg.norm(p)
    if pn != 0 and pn > 1e-5:
        # This prevents the no rotation quaternion [1; 0; 0; 0] and undefined [0; 0; 0; 0]
        # as well as numerically instable quarternions (pn < 1e-5).
        return p / pn

def q2r(q):
    q = qnorm(q)
    o = q[0]
    x = q[1]
    y = q[2]
    z = q[3]
    R = np.array([[o**2 + x**2 - y**2 - z**2,  2 * (x * y - o * z),        2 * (x * z + o * y)],
                  [2 * (x * y + o * z),        o**2 + y**2 - x**2 - z**2,  2 * (y * z - o * x)],
                  [2 * (x * z - o * y),        2 * (y * z + o * x),        o**2 + z**2 - x**2 - y**2]])
    return R

def get_T(R, d):
    T = np.empty((4, 4))
    T[:3, :3] = R
    T[:3, 3] = d
    T[3, :] = [0, 0, 0, 1]
    return T

def compute_RBF(x, y, sigma):

    n, d = x.shape
    m, d = y.shape

    K = np.zeros([n, m])

    for i in range(d):
        K += np.square((x[:, [i]] * np.ones([1, m]) - np.ones([n, 1]) * y[:, i].T))

    K = np.exp(np.divide(K, (-2 * (sigma ** 2))))

    return K

def compute_TPS(y):

    sigma = np.random.normal()
    c = np.random.normal(loc=0, scale=0.1, size=(10, 3))
    x = np.random.uniform(low=-1, high=1, size=(10, 3))

    k = compute_RBF(x, y, sigma)

    return np.matmul(k.T, c) + y

class DataGenerator(Sequence):
    def __init__(self, file_name, batch_size, scale=1, deform=False):
        self.fie_name = file_name
        self.batch_size = batch_size
        self.scale = scale
        self.deform = deform

    def generator(self):
        f = h5py.File(self.fie_name, mode='r')
        nb_sample = f['data'].shape[0]
        while True:
            index = [n for n in range(nb_sample)]
            random.shuffle(index)
            for i in range(nb_sample // self.batch_size):
                batch_start = i * self.batch_size
                batch_end = (i + 1) * self.batch_size
                batch_index = index[batch_start: batch_end]
                X1 = []
                X2 = []
                Y = []
                for j in batch_index:
                    fixed = f['data'][j]
                    fixed = self.scale * fixed[np.random.randint(fixed.shape[0], size=int(fixed.shape[0] * 0.8)), :]
                    moving = f['data'][j]
                    moving = self.scale * moving[np.random.randint(moving.shape[0], size=int(moving.shape[0] * 0.8)), :]
                    if self.deform:
                        moving_deformed = compute_TPS(moving)
                        moving = moving_deformed - np.mean(moving_deformed, axis=0)

                    y, p, r = ypr_rand(-45, 45)
                    R = q2r(qnorm(e2q(y, p, r)))
                    d = d_rand(self.scale * -1, self.scale * 1)
                    T = get_T(R, d)
                    moving_moved = []
                    for point in moving:
                        point_with_1 = np.append(point, 1)
                        new_point_with_1 = np.dot(T, point_with_1)
                        new_point = new_point_with_1[:-1]
                        moving_moved.append(new_point)

                    X1.append(fixed)
                    X2.append(moving_moved)
                    Y.append(fixed)
                yield [np.array(X1), np.array(X2)], np.array(Y)
