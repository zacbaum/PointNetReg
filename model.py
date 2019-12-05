from keras.layers import Conv1D, MaxPooling1D, Flatten, Dropout, Input, BatchNormalization, Dense, GlobalMaxPooling1D
from keras.layers import Reshape, Lambda, concatenate, multiply
from keras.models import Model
from keras.engine.topology import Layer
from keras.utils import multi_gpu_model, plot_model
from keras import backend as K
import numpy as np
import tensorflow as tf


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
		return tf.matmul(inputs[0], inputs[1])

	def compute_output_shape(self, input_shape):
		output_shape = [input_shape[0][0], input_shape[0][1], input_shape[1][-1]]
		return tuple(output_shape)


def PointNet_features(input_len=None):
	input_points = Input(shape=(input_len, 3))
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
	model = Model(inputs=input_points, outputs=global_feature)

	return model

def ConditionalTransformerNet(fixed_len, moving_len, conditional_len, multi_gpu=True, verbose=False):

	fixed = PointNet_features(fixed_len)
	moving = PointNet_features(moving_len)

	combined_inputs = concatenate([fixed.output, moving.output])

	x = Dense(1024, activation='relu')(combined_inputs)
	x = Dropout(0.5)(x)
	x = Dense(1024, activation='relu')(x)
	x = Dropout(0.5)(x)
	x = Dense(512, activation='relu')(x)
	x = Dropout(0.5)(x)
	x = Dense(512, activation='relu')(x)
	x = Dropout(0.5)(x)
	x = Dense(256, activation='relu')(x)
	x = Dense(conditional_len * 3, activation='linear')(x)
	x = Reshape((-1, 3))(x)

	conditional_input = Input(shape=(conditional_len, 3))
	combined_inputs = multiply([x, conditional_input])
	x = Dense(1024, activation='relu')(combined_inputs)
	x = Dropout(0.5)(x)
	x = Dense(1024, activation='relu')(x)
	x = Dropout(0.5)(x)
	x = Dense(512, activation='relu')(x)
	x = Dropout(0.5)(x)
	x = Dense(512, activation='relu')(x)
	x = Dropout(0.5)(x)
	x = Dense(256, activation='relu')(x)
	x = Dense(3, activation='linear')(x)

	model = Model(inputs=[fixed.input, moving.input, conditional_input], outputs=x)

	if verbose: model.summary()
	plot_model(model, show_shapes=True, to_file='model.png')

	if multi_gpu:
		try:
			model = multi_gpu_model(model)	
		except:
			pass
	return model
