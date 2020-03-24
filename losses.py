import numpy as np
import tensorflow as tf
from keras import backend as K
import tensorflow_probability as tfp
tfd = tfp.distributions

def chamfer_distance(y_true, y_pred):
	if K.int_shape(y_pred)[1] == 4:
		y_pred = y_pred[:, :-1]
		y_true = y_true[:, :-1]

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

def variational_distance(A, B, L, sigma=1.0):
	N = tf.cast(K.int_shape(L)[0], dtype=tf.float32)

	# Part 1A - D(a||a')
	a = tf.reduce_sum(tf.square(B), axis=1)
	a = tf.reshape(a, [-1, 1])
	#D_top = tf.sqrt(a - 2 * tf.matmul(B, tf.transpose(B)) + tf.transpose(a))
	D_top = a - 2 * tf.matmul(B, tf.transpose(B)) + tf.transpose(a)
	# Part 1B - SUM( e^(-D(a||a') / N )
	D_top = tf.truediv(tf.square(D_top), (2 * sigma**2))
	D_top = tf.clip_by_value(tf.exp(-D_top), 1e-10, 1e10)
	D_top = tf.truediv(D_top, N)

	# Part 2A - D(a||b)
	a = tf.reduce_sum(tf.square(B), axis=1)
	a = tf.reshape(a, [-1, 1])
	b = tf.reduce_sum(tf.square(A), axis=1)
	b = tf.reshape(b, [1, -1])
	#D_bottom = tf.sqrt(a - 2 * tf.matmul(B, tf.transpose(A)) + b)
	D_bottom = a - 2 * tf.matmul(B, tf.transpose(A)) + b
	# Part 2B - SUM( e^(-D(a||b) / N )
	D_bottom = tf.truediv(tf.square(D_bottom), (2 * sigma**2))
	D_bottom = tf.clip_by_value(tf.exp(-D_bottom), 1e-10, 1e10)
	D_bottom = tf.truediv(D_bottom, N)
	
	# Part 3 - SUM( log( 2A / 2B ) ) / N
	main_div = tf.math.log(tf.reduce_sum(D_top, axis=1)) - tf.math.log(tf.reduce_sum(D_bottom, axis=1))
	main_div = tf.reduce_sum(main_div) / N
	return K.abs(main_div)

def variational_loss(y_true, y_pred):
	batched_losses_AB = tf.map_fn(lambda x: variational_distance(x[0], x[1], x[2]), (y_true, y_pred, y_pred), dtype=tf.float32)
	batched_losses_BA = tf.map_fn(lambda x: variational_distance(x[0], x[1], x[2]), (y_pred, y_true, y_pred), dtype=tf.float32)
	return K.mean(tf.stack([tf.stack(batched_losses_AB),
							tf.stack(batched_losses_BA)]))

def gmm_nll_loss(covariance_matrix_diag, mix_param_val):

	def gmm_nll_batched(y_true, y_pred):

		def gmm_nll(y_true, y_pred):
			if K.int_shape(y_pred)[1] == 4:
				y_pred = y_pred[:, :-1]
				y_true = y_true[:, :-1]

			mix_param = tf.constant(mix_param_val)

			N = K.int_shape(y_pred)[0]
			uniform_pred = tf.constant(1 / N)
			point_prob = tf.fill([N], 1 / N)
			covariance_matrix = tf.constant(np.diag([covariance_matrix_diag, covariance_matrix_diag, covariance_matrix_diag]), dtype=tf.float32)

			mix_gauss_pred = tfd.MixtureSameFamily(
				mixture_distribution=tfd.Categorical(
					probs=point_prob),
				components_distribution=tfd.MultivariateNormalFullCovariance(
					loc=y_pred,
					covariance_matrix=covariance_matrix))

			pdf = mix_gauss_pred.prob(y_true)
			pdf = (mix_param * uniform_pred) + ((1 - mix_param) * pdf)

			return - K.mean(tf.math.log(pdf))

		batched_losses = tf.map_fn(lambda x: gmm_nll(x[0], x[1]), (y_true, y_pred), dtype=tf.float32)
		return K.mean(batched_losses)

	return gmm_nll_batched
	