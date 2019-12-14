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

def nll(y_true, y_pred):
	std = K.std(y_pred)
	likelihood = tfd.Normal(loc=y_pred, scale=std)
	return - K.mean(likelihood.log_prob(y_true), axis=-1)

def sorted_nll(y_true, y_pred):
	y_true_sorted = tf.sort(y_true, axis=1, direction='ASCENDING')
	y_pred_sorted = tf.sort(y_pred, axis=1, direction='ASCENDING')
	std = K.std(y_pred_sorted)
	likelihood = tfd.Normal(loc=y_pred_sorted, scale=std)
	return - K.mean(likelihood.log_prob(y_true_sorted), axis=-1)

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
	return K.mean(likelihood1.kl_divergence(likelihood2))