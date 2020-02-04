import tensorflow as tf
from keras import backend as K
import numpy as np
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('AGG')
import os
os.environ["CUDA_VISIBLE_DEVICES"]='2'

from data_loader import DataGenerator
from keras.models import load_model
from keras.engine.topology import Layer

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

prostate_data = sio.loadmat('prostate.mat')

all_prostates = []
dims = [1024, 3]

for prostate in prostate_data['PointSets'][0]: # Gets to the list of structs.
	prostate_j = [np.array([])] * 5
	ref_data = []
	for i in range(prostate.shape[1]): # Gets to the individual dataset and gets the reference normalization parameters.

		if prostate[0, i][0].size > 0:
			if prostate[0, i][0] == 'ROI 1':
				mc_data = prostate[0, i][1][:3,:].T
				norm_data = mc_data - np.mean(mc_data, axis=0)

	for j in range(prostate.shape[1]): # Gets to the individual dataset and stores rescaled, upsampled and centered points.
		
		if prostate[0, j][0].size > 0:

			prostate_j_data = prostate[0, j][1][:3,:].T
			prostate_j_data_centered = prostate_j_data - np.mean(mc_data, axis=0)
			prostate_j_data_normalized = 2 * (prostate_j_data_centered - np.min(norm_data)) / np.ptp(norm_data) - 1
			prostate_j_data_resized = np.resize(prostate_j_data_normalized, dims)

			if prostate[0, j][0] == 'ROI 1':
				prostate_j[0] = prostate_j_data_resized
			elif prostate[0, j][0] == 'ROI 2':
				prostate_j[1] = prostate_j_data_resized
			elif prostate[0, j][0] == 'ROI 3':
				prostate_j[2] = prostate_j_data_resized
			elif prostate[0, j][0] == 'ROI 4':
				prostate_j[3] = prostate_j_data_resized
			elif prostate[0, j][0] == 'ROI 5':
				prostate_j[4] = prostate_j_data_resized

	all_prostates.append(prostate_j)

model = load_model('model.h5',
				   custom_objects={'MatMul':MatMul},
				   compile=False)

for i in range(len(all_prostates)):

	if all_prostates[i][2].size == 0: continue

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	fixed = all_prostates[i][0]
	x_fx = [i[0] for i in fixed]
	y_fx = [i[1] for i in fixed]
	z_fx = [i[2] for i in fixed]

	move = all_prostates[i+1][0]
	x_mv = [i[0] for i in move]
	y_mv = [i[1] for i in move]
	z_mv = [i[2] for i in move]

	inp = all_prostates[i+1][0]
	x_inp = [i[0] for i in inp]
	y_inp = [i[1] for i in inp]
	z_inp = [i[2] for i in inp]

	pred = model.predict([[np.array(fixed)],	# Fixed
						  [np.array(move)],		# Moving
						  [np.array(inp)]])		# To_Reg

	x_pred = [i[0] for i in pred[0]]
	y_pred = [i[1] for i in pred[0]]
	z_pred = [i[2] for i in pred[0]]

	ax.scatter(x_fx, y_fx, z_fx, c='y', marker='.')
	#ax.scatter(x_mv, y_mv, z_mv, c='r', marker='.')
	ax.scatter(x_inp, y_inp, z_inp, c='b', marker='.')
	ax.scatter(x_pred, y_pred, z_pred, c='g', marker='.')

	lim = 1
	ax.set_xlim([-lim, lim])
	ax.set_ylim([-lim, lim])
	ax.set_zlim([-lim, lim])

	plt.show()
	plt.savefig('scatter' + str(i) + '.png', dpi=250)
	plt.close()

print("DONE")