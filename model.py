import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Input, BatchNormalization, Dense, RepeatVector, Reshape, Lambda, concatenate, add
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from keras import backend as K
import numpy as np
import tensorflow.keras

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

	x = Dense(dimensions * dimensions, weights=[np.zeros([256, dimensions * dimensions]), np.eye(dimensions).flatten().astype(np.float32)])(x)
	input_T = Reshape((dimensions, dimensions))(x)

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
	global_feature = Reshape((1024,))(global_feature)

	model = Model(inputs=input_points, outputs=global_feature, name='PointNet')

	return model

def TPSTransformNet(num_points, dims=3, tps_features=27, sigma=1.0, ct_activation='relu', dropout=0., batch_norm=False, verbose=False):

	def tps(inputs):
		return tf.map_fn(lambda x: register_tps(x[0], x[1]), inputs)

	def register_tps(inputs, y):

		x = tf.slice(inputs, [0], [tf.shape(inputs)[0]])
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

	fixed = Input(shape=(num_points, dims), name='Fixed_Model')
	moved = Input(shape=(num_points, dims), name='Moved_Model')
	moving = Input(shape=(num_points, dims), name='Moving_Model')

	pointNet = PointNet_features(num_points, dims)
	fixed_pointNet = pointNet(fixed)
	moving_pointNet = pointNet(moved)

	point_features = concatenate([fixed_pointNet, moving_pointNet])

	nodes = [1024, 1024, 512, 512]
	for num_nodes in nodes:
		point_features = Dense(num_nodes, activation=ct_activation)(point_features)
		if dropout:
			point_features = Dropout(dropout)(point_features)
		if batch_norm:
			point_features = BatchNormalization()(point_features)

	point_features = Dense(tps_features * dims * 2)(point_features)

	x = Lambda(tps, name='TPS_Registration')([point_features, moving])
	x = add(x)

	model = Model(inputs=[fixed, moved, moving], outputs=x)

	if verbose: model.summary()
	plot_model(model, to_file='model.png', show_shapes=True)#, expand_nested=True)

	return model

def ConditionalTransformerNet(num_points, dims=3, ct_activation='relu', dropout=0., batch_norm=False, verbose=False):

	def drop_batch_points(batch):
		return tf.map_fn(lambda x: drop_points(x), batch)

	def drop_points(points):
		shuffled = tf.random.shuffle(points)
		subsets = tf.split(shuffled, 4)
		return subsets[0]

	fixed = Input(shape=(num_points, dims), name='Fixed_Model')
	fixed_ss = Lambda(drop_batch_points, name='Fixed_Subsampled')(fixed)
	moved = Input(shape=(num_points, dims), name='Moved_Model')
	moved_ss = Lambda(drop_batch_points, name='Moved_Subsampled')(moved)
	moving = Input(shape=(num_points, dims), name='Moving_Model')

	pointNet = PointNet_features(int(num_points / 4), dims)
	fixed_pointNet = pointNet(fixed_ss)
	moving_pointNet = pointNet(moved_ss)

	point_features = concatenate([fixed_pointNet, moving_pointNet])
	point_features_matrix = RepeatVector(num_points)(point_features)
	x = concatenate([point_features_matrix, moving])

	filters = [1024, 512, 256, 128, 64]
	for num_filters in filters:
		x = Conv1D(num_filters, 1, activation=ct_activation)(x)
		if dropout:
			x = Dropout(dropout)(x)
		if batch_norm:
			x = BatchNormalization()(x)
	x = Conv1D(dims, 1)(x)
	x = add([x, moving])

	model = Model(inputs=[fixed, moved, moving], outputs=x, name='CTN')

	if verbose: model.summary()
	plot_model(model, to_file='model.png', show_shapes=True)#, expand_nested=True)

	return model