import numpy as np
import scipy
import h5py
import random
from keras.utils import np_utils
from keras.utils import Sequence
import random
from itertools import product

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
    m, _ = y.shape
    K = np.zeros([n, m])
    for i in range(d):
        K += np.square((x[:, [i]] * np.ones([1, m]) - np.ones([n, 1]) * y[:, i].T))
    K = np.exp(np.divide(K, (-2 * (sigma ** 2))))

    return K

def compute_RBF_defm(y):

    sigma = np.random.normal()
    c = np.random.normal(loc=0, scale=0.25, size=(8, 3))
    x = np.array(list(product(range(-1, 2, 2), repeat=3)))
    k = compute_RBF(x, y, sigma)

    return np.matmul(k.T, c) + y

def compute_EBS_gauss(p, x, sigma, nu):

    n, d = p.shape
    m, _ = x.shape
    xd = np.zeros((m, n, 3))
    for i in range(d):
        xd[:, :, i] = np.ones((m, 1)) * p[:, i].T - np.reshape(x[:, i], [-1, 1]) * np.ones((1, n))
    K = np.zeros((m*3, n*3))
    for row in range(m):
        for col in range(n):
            K_r_c = basis_gauss(xd[row, col, :], d, sigma, nu)
            K[3*row:3*row+3, 3*col:3*col+3] = K_r_c
    
    return K

def basis_gauss(y, d, sigma, nu):

    sigma2 = sigma**2
    r2 = np.matmul(y.T, y)
    if r2 == 0: r2 = 1e-8
    r = np.sqrt(r2)
    rhat = r / (np.sqrt(2) * sigma)
    c1 = scipy.special.erf(rhat) / r
    c2 = np.sqrt(2 / np.pi) * sigma * np.exp(-rhat**2) / r2
    g = ((4 * (1 - nu) - 1) * c1 - c2 + sigma2 * c1 / r2) * np.eye(d) + (c1 / r2 + 3 * c2 / r2 - 3 * sigma2 * c1 / (r2 * r2)) * (y * np.reshape(y, [-1, 1]))
    
    return g

def compute_EBS_w(p, q, sigma, nu, lmda=1e-6):
    
    n, d = p.shape
    m, _ = q.shape
    lmda = lmda * np.eye(n * d)
    k = compute_EBS_gauss(p, p, sigma, nu)
    y = np.reshape((q - p).T, [n * d, 1], order='F')
    L = k + lmda
    U, S, V = np.linalg.svd(L)
    w = np.matmul(np.matmul(np.matmul(U, np.diag(np.reciprocal(S))), V), y)
    
    return w

def compute_EBS_gauss_defm(x, sigma=0.1, nu=0.1):

    p = x[np.random.randint(x.shape[0], size=10), :]
    q = p + np.random.uniform(low=-0.1, high=0.1, size=(10, 3))
    n, d = p.shape
    m, _ = x.shape
    w = compute_EBS_w(p, q, sigma, nu)
    k = compute_EBS_gauss(p, x, sigma, nu)

    return np.reshape(np.matmul(k, w), [d, m]).T + x

class DataGenerator(Sequence):
    def __init__(self, file_name, batch_size, scale=1, deform=False, part=0):
        self.fie_name = file_name
        self.batch_size = batch_size
        self.scale = scale
        self.deform = deform
        self.part = part

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
                X3 = []
                Y = []
                for j in batch_index:
                    # Extract data.
                    fixed = f['data'][j]
                    moving = f['data'][j]
                    dims = fixed.shape
                    
                    # Normalize between [-1, 1] and mean center.
                    fixed = 2 * (fixed - np.min(fixed)) / np.ptp(fixed) - 1
                    fixed = fixed - np.mean(fixed, axis=0)

                    moving = 2 * (moving - np.min(moving)) / np.ptp(moving) - 1
                    moving = moving - np.mean(moving, axis=0)

                    # Deform and recenter.
                    if self.deform:
                        moving_deformed = compute_RBF_defm(moving)
                        moving = moving_deformed - np.mean(moving_deformed, axis=0)

                    # Rotate, translate.
                    y, p, r = ypr_rand(-45, 45)
                    R = q2r(qnorm(e2q(y, p, r)))
                    d = d_rand(self.scale * -1, self.scale * 1)
                    T = get_T(R, d)
                    
                    moving_with_ones = np.ones((dims[0], dims[1] + 1))
                    moving_with_ones[:,:-1] = moving
                    moving_moved = np.dot(T, moving_with_ones.T).T[:, :-1]

                    # Recenter again.
                    moving_moved = moving_moved - np.mean(moving_moved, axis=0)

                    # Add some noise to the points.
                    to_reg = moving_moved

                    # Take part(s) from point set(s).
                    if self.part > 0: # Register a part to whole
                        axis = np.random.randint(0, 3)
                        moving_moved = moving_moved[moving_moved[:, axis].argsort()]
                        moving_moved = moving_moved[int(0.5 * dims[0]):]
                        moving_moved = np.resize(moving_moved, dims)
                        if self.part > 1: # Register a part to a part
                            axis = np.random.randint(0, 3)
                            fixed = fixed[fixed[:, axis].argsort()]
                            fixed = fixed[:int(0.5 * dims[0])]
                            fixed = np.resize(fixed, dims)

                    moving_moved = moving_moved + np.random.normal(0, 5e-2, dims)
                    
                    X1.append(fixed)
                    X2.append(moving_moved)
                    X3.append(to_reg)
                    Y.append(fixed)
                yield [np.array(X1), np.array(X2), np.array(X3)], np.array(Y)
                