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


def chamfer_distance(y_true, y_pred):
	row_norms_true = tf.reduce_sum(tf.square(y_true), axis=1)
	row_norms_true = tf.reshape(row_norms_true, [-1, 1])
	row_norms_pred = tf.reduce_sum(tf.square(y_pred), axis=1)
	row_norms_pred = tf.reshape(row_norms_pred, [1, -1])
	D = row_norms_true - 2 * tf.matmul(y_true, y_pred, False, True) + row_norms_pred
	dist_t_to_p = K.mean(K.min(D, axis=0))
	dist_p_to_t = K.mean(K.min(D, axis=1))
	dist = K.mean(tf.stack([dist_p_to_t, dist_t_to_p]))
	return dist

def chamfer_loss(y_true, y_pred):
	batched_losses = tf.map_fn(lambda x: chamfer_distance(x[0], x[1]), (y_true, y_pred), dtype=tf.float32)
	return K.mean(tf.stack(batched_losses))


def gmm_nll_loss(y_true, y_pred):

	def gmm_nll(y_true, y_pred):
		
		mix_parameter = tf.fill([K.int_shape(y_pred)[0]], 1 / K.int_shape(y_pred)[0])
		covariance_matrix = np.diag([1e-4, 1e-4, 1e-4])
		covariance_matrix = tf.constant(covariance_matrix, dtype=tf.float32)

		mix_gauss_pred = tfd.MixtureSameFamily(
			mixture_distribution=tfd.Categorical(
				probs=mix_parameter),
			components_distribution=tfd.MultivariateNormalFullCovariance(
				loc=y_pred,
				covariance_matrix=covariance_matrix))

		return - K.mean(mix_gauss_pred.log_prob(y_true))

	batched_losses = tf.map_fn(lambda x: gmm_nll(x[0], x[1]), (y_true, y_pred), dtype=tf.float32)
	return K.mean(batched_losses)