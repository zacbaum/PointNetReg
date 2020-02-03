from keras.layers import Conv1D, MaxPooling1D, Dropout, Input, BatchNormalization, Dense, RepeatVector, GaussianNoise
from keras.layers import Reshape, Lambda, concatenate, add
from keras.models import Model
from keras.engine.topology import Layer
from keras.utils import multi_gpu_model, plot_model
from keras import backend as K
import numpy as np
import tensorflow as tf
import keras

class MatMul(Layer):

	def __init__(self, **kwargs):
		super(MatMul, self).__init__(**kwargs)

	def build(self, input_shape):
		# Used purely for shape validation.
		if not isinstance(input_shape, list):
			raise ValueError('`MatMul` layer should be called '
							 'on a list of inputs')
		if len(input_shape) != 2:
			raise ValueError('The input of `MatMul` layer should be a list containing 2 elements')

		if len(input_shape[0]) != 3 or len(input_shape[1]) != 3:
			raise ValueError('The dimensions of each element of inputs should be 3')

		if input_shape[0][-1] != input_shape[1][1]:
			raise ValueError('The last dimension of inputs[0] should match the dimension 1 of inputs[1]')

	def call(self, inputs):
		if not isinstance(inputs, list):
			raise ValueError('A `MatMul` layer should be called '
							 'on a list of inputs.')
		import tensorflow as tf
		return tf.matmul(inputs[0], inputs[1])

	def compute_output_shape(self, input_shape):
		output_shape = [input_shape[0][0], input_shape[0][1], input_shape[1][-1]]
		return tuple(output_shape)

def PointNet_features(input_len, dimensions=3):

	input_points = Input(shape=(input_len, dimensions))
	# input transformation net
	x = Conv1D(64, 1, activation='relu')(input_points)
	x = BatchNormalization()(x)
	x = Conv1D(128, 1, activation='relu')(x)
	x = BatchNormalization()(x)
	x = Conv1D(1024, 1, activation='relu')(x)
	x = BatchNormalization()(x)
	x = MaxPooling1D(pool_size=input_len)(x)

	x = Dense(512, activation='relu')(x)
	x = BatchNormalization()(x)
	x = Dense(256, activation='relu')(x)
	x = BatchNormalization()(x)

	x = Dense(9, weights=[np.zeros([256, 9]), np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)])(x)
	input_T = Reshape((3, 3))(x)

	# forward net
	g = MatMul()([input_points, input_T])
	g = Conv1D(64, 1, activation='relu')(g)
	g = BatchNormalization()(g)
	g = Conv1D(64, 1, activation='relu')(g)
	g = BatchNormalization()(g)

	# feature transform net
	f = Conv1D(64, 1, activation='relu')(g)
	f = BatchNormalization()(f)
	f = Conv1D(128, 1, activation='relu')(f)
	f = BatchNormalization()(f)
	f = Conv1D(1024, 1, activation='relu')(f)
	f = BatchNormalization()(f)
	f = MaxPooling1D(pool_size=input_len)(f)
	f = Dense(512, activation='relu')(f)
	f = BatchNormalization()(f)
	f = Dense(256, activation='relu')(f)
	f = BatchNormalization()(f)
	f = Dense(64 * 64, weights=[np.zeros([256, 64 * 64]), np.eye(64).flatten().astype(np.float32)])(f)
	feature_T = Reshape((64, 64))(f)

	# forward net
	g = MatMul()([g, feature_T])
	g = Conv1D(64, 1, activation='relu')(g)
	g = BatchNormalization()(g)
	g = Conv1D(128, 1, activation='relu')(g)
	g = BatchNormalization()(g)
	g = Conv1D(1024, 1, activation='relu')(g)
	g = BatchNormalization()(g)

	# global feature
	global_feature = MaxPooling1D(pool_size=input_len)(g)
	global_feature = Reshape((-1,))(global_feature)

	model = Model(inputs=input_points, outputs=global_feature, name='PointNet')

	return model

def TPSTransformNet(num_points, dimensions=3, tps_features=1000, ct_initializer='he_uniform', ct_activation='relu', dropout=0., multi_gpu=True, verbose=False):

	def mean_subtract(input_tensor):
		import tensorflow as tf
		return tf.map_fn(lambda x: x - tf.reduce_mean(x, axis=0), input_tensor)

	def tps(inputs):
		return tf.map_fn(lambda x: register_tps(x[0], x[1]), inputs)

	def register_tps(inputs, y):

		sigma = tf.slice(inputs, [tf.shape(inputs)[0] - 1], [1])
		x = tf.slice(inputs, [0], [tf.shape(inputs)[0] - 1])
		x = tf.reshape(x, [2, -1, 3])

		c = x[0]
		x = x[1]

		x_norms = tf.reduce_sum(tf.square(x), axis=1)
		x_norms = tf.reshape(x_norms, [-1, 1])

		y_norms = tf.reduce_sum(tf.square(y), axis=1)
		y_norms = tf.reshape(y_norms, [-1, 1])

		k1 = x_norms * tf.ones([1, K.int_shape(y)[0]])
		k2 = tf.ones([K.int_shape(x)[0], 1]) * tf.transpose(y_norms)

		k = k1 + k2
		k -= (2 * tf.matmul(x, y, False, True))
		k = tf.exp(tf.truediv(k, (-2 * tf.square(sigma))))

		x0 = tf.matmul(k, c, True, False)

		return [x0, y]

	pointNet = PointNet_features(num_points, dimensions)

	fixed = Input(shape=(num_points, dimensions), name='Fixed_Model')
	moving = Input(shape=(num_points, dimensions), name='Moving_Model')
	moving_mean_subtracted = Lambda(mean_subtract, name='Mean_Subtraction')(moving)

	fixed_pointNet = pointNet(fixed)
	moving_pointNet = pointNet(moving_mean_subtracted)

	point_features = concatenate([fixed_pointNet, moving_pointNet])

	x = Dense(tps_features, kernel_initializer=ct_initializer, activation=ct_activation)(point_features)
	if dropout > 0:
		x = Dropout(dropout)(x)
	x = BatchNormalization()(x)
	
	x = Dense(tps_features, kernel_initializer=ct_initializer, activation=ct_activation)(x)
	if dropout > 0:
		x = Dropout(dropout)(x)
	x = BatchNormalization()(x)
	
	x = Dense(tps_features * dimensions, kernel_initializer=ct_initializer, activation=ct_activation)(x)
	if dropout > 0:
		x = Dropout(dropout)(x)
	x = BatchNormalization()(x)

	x = Dense(tps_features * dimensions, kernel_initializer=ct_initializer, activation=ct_activation)(x)
	if dropout > 0:
		x = Dropout(dropout)(x)
	x = BatchNormalization()(x)

	x = Dense(tps_features * dimensions * 2, kernel_initializer=ct_initializer, activation=ct_activation)(x)
	if dropout > 0:
		x = Dropout(dropout)(x)
	x = BatchNormalization()(x)

	x = Dense(tps_features * dimensions * 2, kernel_initializer=ct_initializer, activation=ct_activation)(x)
	if dropout > 0:
		x = Dropout(dropout)(x)
	x = BatchNormalization()(x)

	x = Dense(tps_features * dimensions * 2 + 1, kernel_initializer=ct_initializer)(x)
	x = BatchNormalization()(x)
	
	x = Lambda(tps, name='TPS_Registration')([x, moving_mean_subtracted])
	x = add(x)

	model = Model(inputs=[fixed, moving], outputs=x)

	if verbose: model.summary()
	plot_model(model, to_file='model.png', show_shapes=True)#, expand_nested=True)

	if multi_gpu:
		try:
			model = multi_gpu_model(model)	
		except:
			pass
	return model

def ConditionalTransformerNet(num_points, dimensions=3, ct_activation='relu', dropout=0., batch_norm=False, noise=0, multi_gpu=True, verbose=False):

	def mean_subtract(input_tensor):
		import tensorflow as tf
		return tf.map_fn(lambda x: x - tf.reduce_mean(x, axis=0), input_tensor)

	fixed = Input(shape=(num_points, dimensions), name='Fixed_Model')
	moved = Input(shape=(num_points, dimensions), name='Moved_Model')

	moving = Input(shape=(num_points, dimensions), name='Moving_Model')

	pointNet = PointNet_features(num_points, dimensions)
	fixed_pointNet = pointNet(fixed)
	moving_pointNet = pointNet(moved)

	point_features = concatenate([fixed_pointNet, moving_pointNet])
	point_features_matrix = RepeatVector(num_points)(point_features)
	
	x = concatenate([point_features_matrix, moving])

	#filters = [1024, 256, 64, dimensions]
	filters = [1024, 512, 256, 128, 64, dimensions]
	#filters = [2048, 1024, 1024, 512, 512, 256, 256, 128, 128, 64, 64, dimensions]
	for num_filters in filters:
		if num_filters == dimensions:
			x = Conv1D(num_filters, 1)(x)
		else:
			x = Conv1D(num_filters, 1, activation=ct_activation)(x)
			if dropout:
				x = Dropout(dropout)(x)
			if batch_norm:
				x = BatchNormalization()(x)

	x = add([x, moving])
	if noise:
		x = GaussianNoise(noise)(x)

	model = Model(inputs=[fixed, moved, moving], outputs=x)

	if verbose: model.summary()
	plot_model(model, to_file='model.png', show_shapes=True)#, expand_nested=True)

	if multi_gpu:
		try:
			model = multi_gpu_model(model)	
		except:
			pass
	return model
