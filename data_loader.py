import numpy as np
import scipy
from scipy.spatial import distance 
import h5py
import random
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import random
from itertools import product

class DataGenerator(Sequence):
	def __init__(self, data, batch_size, dims=3, shuffle=True, rotate=45, displace=1, deform=False, part=0, part_nn=0):
		self.data = data
		self.batch_size = batch_size
		self.dims = dims

		self.shuffle = shuffle
		self.rotate = rotate
		self.displace = displace
		self.deform = deform
		self.part = part
		self.part_nn = part_nn

		self.nb_sample = self.data['data'].shape[0]
		self.indexes = np.arange(self.nb_sample)

	def __len__(self):
		return int(np.floor(self.nb_sample / self.batch_size))

	def __getitem__(self, index):
		indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

		# Find list of IDs
		data_temp = [self.data['data'][k] for k in indexes]

		# Generate augmented batch data
		X, y = self.__generator(data_temp)

		return X, y

	def on_epoch_end(self):
		if self.shuffle:
			np.random.shuffle(self.indexes)

	def __generator(self, data):
		X1 = []
		X2 = []
		X3 = []
		Y = []
		eps = 1e-7
		for d in data:
			# Extract data.
			fixed = d
			dims = fixed.shape

			# Normalize between [-1, 1] and mean center.
			fixed = 2 * (fixed - np.min(fixed)) / (np.ptp(fixed) + eps) - 1
			fixed = fixed - np.mean(fixed, axis=0)
			moving = fixed
			ground_truth = moving

			# Take part(s) from point set(s).
			if not self.part_nn:
				if self.part > 0: # Register a part to whole
					axis_moving = np.random.randint(0, 3)
					moving = moving[moving[:, axis_moving].argsort()]
					moving = moving[int(0.5 * dims[0]):]
					moving = np.resize(moving, dims)
					if self.part > 1: # Register a part to a part
						axis_fixed = np.random.randint(0, 3)
						while axis_moving == axis_fixed:
							axis_fixed = np.random.randint(0, 3)
						fixed = fixed[fixed[:, axis_fixed].argsort()]
						fixed = fixed[:int(0.5 * dims[0])]
						fixed = np.resize(fixed, dims)
			if self.part_nn:
				if self.part > 0: # Register a part to whole
					index = np.random.randint(0, dims[0])
					D = distance.cdist([moving[index, :]], moving)
					closest = np.argsort(D[0])
					closest = closest[:self.part_nn]
					moving = moving[closest, :]
					moving = np.resize(moving, dims)
					if self.part > 1: # Register a part to a part
						index = np.random.randint(0, dims[0])
						D = distance.cdist([moving[index, :]], moving)
						closest = np.argsort(D[0])
						closest = closest[:self.part_nn]
						fixed = fixed[closest, :]
						fixed = np.resize(fixed, dims)
			
			ground_truth = moving # Make the ground truth the unmoved-moving part.

			# Deform.
			if self.deform:
				sigma = np.random.normal()
				c = np.random.normal(loc=0, scale=0.1, size=(10, 3))
				x = np.random.uniform(low=-1, high=1, size=(10, 3))
				k = compute_RBF(x, moving, sigma)
				moving = np.matmul(k.T, c) + moving
				moving_mean = np.mean(moving, axis=0)
				moving = moving - moving_mean

			# Rotate, translate.
			y, p, r = ypr_rand(-self.rotate, self.rotate)	
			R = e2r(y, p, r)
			d = d_rand(-self.displace, self.displace)
			T = get_T(R, d)
			moving_with_ones = np.ones((dims[0], dims[1] + 1))
			moving_with_ones[:,:-1] = moving
			moving = np.dot(T, moving_with_ones.T).T[:, :-1]
			to_reg = moving
			
			if self.dims == 4:
				moving_with_ones = np.ones((dims[0], dims[1] + 1))
				moving_with_ones[:,:-1] = moving
				moving = moving_with_ones
				to_reg_with_ones = np.ones((dims[0], dims[1] + 1))
				to_reg_with_ones[:,:-1] = to_reg
				to_reg = to_reg_with_ones
				fixed_with_ones = np.ones((dims[0], dims[1] + 1))
				fixed_with_ones[:,:-1] = fixed
				fixed = fixed_with_ones
				gt_with_ones = np.ones((dims[0], dims[1] + 1))
				gt_with_ones[:,:-1] = ground_truth
				ground_truth = gt_with_ones

			X1.append(fixed)
			X2.append(moving)
			X3.append(to_reg)
			Y.append(ground_truth)
		return [np.array(X1), np.array(X2), np.array(X3)], np.array(Y)
				

def ypr_rand(min_deg, max_deg):
	y = [random.uniform(min_deg, max_deg)]
	p = [random.uniform(min_deg, max_deg)]
	r = [random.uniform(min_deg, max_deg)]
	return y, p, r

def d_rand(min_d, max_d):
	d = [random.uniform(min_d, max_d), random.uniform(min_d, max_d), random.uniform(min_d, max_d)]
	return d

def e2r(yaw, pitch, roll):

	yaw_rad = (np.deg2rad(yaw) / 2)
	pitch_rad = (np.deg2rad(pitch) / 2)
	roll_rad = (np.deg2rad(roll) / 2)
	R_x = np.array([[1,		0,                 	0               ],
					[0,		np.cos(roll_rad),	-np.sin(roll_rad)],
					[0,		np.sin(roll_rad),	np.cos(roll_rad) ]])
	R_y = np.array([[np.cos(pitch_rad),		0,      np.sin(pitch_rad)],
					[0,						1,      0                ],
					[-np.sin(pitch_rad),  	0,		np.cos(pitch_rad)]])
	R_z = np.array([[np.cos(yaw_rad),		-np.sin(yaw_rad),		0],
					[np.sin(yaw_rad),		np.cos(yaw_rad),		0],
					[0,						0,                      1]])
	R = np.dot(R_z, np.dot(R_y, R_x))
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
	c = np.random.normal(loc=0, scale=0.1, size=(10, 3))
	x = np.random.uniform(low=-1, high=1, size=(10, 3))
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