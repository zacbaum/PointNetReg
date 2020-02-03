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

for prostate in prostate_data['PointSets'][0]: # Gets to the list of structs.
	prostate_i = [np.array([])] * 5
	for i in range(prostate.shape[1]): # Gets to the individual dataset.
		if prostate[0, i][0] == 'ROI 1':
			prostate_i[0] = prostate[0, i][1][:3,:].T
		elif prostate[0, i][0] == 'ROI 2':
			prostate_i[1] = prostate[0, i][1][:3,:].T
		elif prostate[0, i][0] == 'ROI 3':
			prostate_i[2] = prostate[0, i][1][:3,:].T
		elif prostate[0, i][0] == 'ROI 4':
			prostate_i[3] = prostate[0, i][1][:3,:].T
		elif prostate[0, i][0] == 'ROI 5':
			prostate_i[4] = prostate[0, i][1][:3,:].T
	all_prostates.append(prostate_i)

# SHOULD WE BE CONCATENATING SURFACE WITH TRANSITION?
# SHOULD WE BE CONCATENATING ROIS?

# UPSAMPLE ALL TO 1024 POINTS

model = load_model('model.h5',
				   custom_objects={'MatMul':MatMul},
				   compile=False)

test_file = './ModelNet40/ply_data_test.h5'
scale = 1
test = DataGenerator(test_file, 1, scale=scale, deform=True, part=True)

for it in range(10):
	test_data = []     # store all the generated data batches
	test_labels = []   # store all the generated ground_truth batches
	max_iter = 1        # maximum number of iterations, in each iteration one batch is generated; the proper value depends on batch size and size of whole data
	i = 0
	for d, l in test.generator():
		test_data.append(d)
		test_labels.append(l)
		i += 1
		if i == max_iter:
			break

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	fixed = test_data[0][0][0]
	x_fx = [i[0] for i in fixed]
	y_fx = [i[1] for i in fixed]
	z_fx = [i[2] for i in fixed]

	defm = test_data[0][1][0]
	x_def = [i[0] for i in defm]
	y_def = [i[1] for i in defm]
	z_def = [i[2] for i in defm]

	inp = test_data[0][2][0]
	x_inp = [i[0] for i in inp]
	y_inp = [i[1] for i in inp]
	z_inp = [i[2] for i in inp]

	pred = model.predict(test_data[0])

	x_pred = [i[0] for i in pred[0]]
	y_pred = [i[1] for i in pred[0]]
	z_pred = [i[2] for i in pred[0]]

	ax.scatter(x_fx, y_fx, z_fx, c='y', marker='.')
	ax.scatter(x_def, y_def, z_def, c='r', marker='.')
	#ax.scatter(x_inp, y_inp, z_inp, c='b', marker='.')
	ax.scatter(x_pred, y_pred, z_pred, c='g', marker='.')

	lim = 1 * scale
	ax.set_xlim([-lim, lim])
	ax.set_ylim([-lim, lim])
	ax.set_zlim([-lim, lim])

	plt.show()
	plt.savefig('scatter' + str(it) + '.png', dpi=250)
	plt.close()
'''