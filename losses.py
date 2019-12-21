import numpy as np
from keras import backend as K
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

def sorted_mse_loss(y_true, y_pred):
	y_true_sorted = tf.sort(y_true, axis=1, direction='ASCENDING')
	y_pred_sorted = tf.sort(y_pred, axis=1, direction='ASCENDING')
	y_true_sorted = K.cast(y_true_sorted, y_pred_sorted.dtype)
	return K.mean(K.square(y_true_sorted - y_pred_sorted), axis=-1)


def sorted_mse_loss2(y_true, y_pred):

	def sorted_mse(y_true, y_pred):

		def euclidean_sort(T, P):
			newP = np.array([])
			for i in range(len(T)):
				current_closest = [float('Inf'), float('Inf'), float('Inf')]
				closest_index = -1
				for j in range(len(P)):
					if np.linalg.norm(P[j] - T[i]) < np.linalg.norm(current_closest - T[i]):
						current_closest = P[j]
						closest_index = j
				newP = np.vstack([newP, current_closest]) if newP.size else current_closest
				P = np.delete(P, closest_index, axis=0)
			return newP

		y_pred = tf.py_func(euclidean_sort, [y_true, y_pred], tf.float32)
		return K.mean(K.square(y_true - y_pred))

	batched_losses = tf.map_fn(lambda x: sorted_mse(x[0], x[1]), (y_true, y_pred), dtype=tf.float32)
	return K.mean(tf.stack(batched_losses))


def nll(y_true, y_pred):
	std = K.std(y_pred)
	likelihood = tfd.Normal(loc=y_pred, scale=std)
	return - K.mean(likelihood.log_prob(y_true), axis=-1)


def kl_divergence(y_true, y_pred):
	std = K.std(y_true)
	likelihood1 = tfd.Normal(loc=y_true, scale=std)
	std = K.std(y_pred)
	likelihood2 = tfd.Normal(loc=y_pred, scale=std)
	return K.mean(likelihood1.kl_divergence(likelihood2))


def sorted_kl_divergence(y_true, y_pred):
	y_true_sorted = tf.sort(y_true, axis=1, direction='ASCENDING')
	y_pred_sorted = tf.sort(y_pred, axis=1, direction='ASCENDING')
	std = K.std(y_true_sorted)
	likelihood1 = tfd.Normal(loc=y_true_sorted, scale=std)
	std = K.std(y_pred_sorted)
	likelihood2 = tfd.Normal(loc=y_pred_sorted, scale=std)
	l1 = K.mean(likelihood1.kl_divergence(likelihood2))
	l2 = K.mean(likelihood2.kl_divergence(likelihood1))
	return K.mean(tf.stack([l1, l2]))


def chamfer_distance(y_true, y_pred):
	row_norms_true = tf.reduce_sum(tf.square(y_true), axis=1)
	row_norms_true = tf.reshape(row_norms_true, [-1, 1])
	row_norms_pred = tf.reduce_sum(tf.square(y_pred), axis=1)
	row_norms_pred = tf.reshape(row_norms_pred, [1, -1])
	D = tf.sqrt(tf.maximum(row_norms_true - 2 * tf.matmul(y_true, y_pred, False, True) + row_norms_pred, 0.0))
	dist_t_to_p = K.mean(K.min(D, axis=0)) #shape: (1,)
	dist_p_to_t = K.mean(K.min(D, axis=1)) #shape: (1,)
	dist = K.max([dist_p_to_t, dist_t_to_p]) #shape: (1,)
	return dist

def chamfer_loss(y_true, y_pred):
	batched_losses = tf.map_fn(lambda x: chamfer_distance(x[0], x[1]), (y_true, y_pred), dtype=tf.float32)
	return K.mean(tf.stack(batched_losses))