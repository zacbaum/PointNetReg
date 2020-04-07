import tensorflow as tf
from keras import backend as K
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('AGG')
import os
os.environ["CUDA_VISIBLE_DEVICES"]='CPU'
from model import MatMul, ConditionalTransformerNet
import h5py
from data_loader import DataGenerator

try:
	v = int(tf.VERSION[0])
except AttributeError:
	v = int(tf.__version__[0])

if v >= 2:
	from tensorflow.keras.models import load_model
	from tensorflow.keras.layers import Layer
else:
	from keras.models import load_model
	from keras.engine.topology import Layer

def chamfer(a, b):
	D = scipy.spatial.distance.cdist(a, b)
	return np.mean([np.mean(np.min(D, axis=0)), np.mean(np.min(D, axis=1))])

def hausdorff_2way(a, b):
	D = scipy.spatial.distance.cdist(a, b)
	return np.max([np.max(np.min(D, axis=0)), np.max(np.min(D, axis=1))])

def predict_mn40(fname, rotate, displace, deform):
	if not os.path.exists('./prediction_results-' + fname + '/'):
			os.mkdir('./prediction_results-' + fname + '/')

	test_file = './ModelNet40/ply_data_test.h5'
	test = h5py.File(test_file, mode='r')
	num_val = test['data'].shape[0]

	batch_size = 32

	val = DataGenerator(test,
						batch_size,
						rotate=rotate,
						displace=displace,
						deform=deform,
						dims=4,
						part=1)

	val_data = []
	val_labels = []
	max_iter = num_val // batch_size
	i = 0
	for d, l in val:
		val_data.append(d)
		val_labels.append(l)
		i += 1
		if i == max_iter:
			break

	# Load the model.
	model = load_model(fname + '.h5',
					   custom_objects={'MatMul':MatMul}, 
					   compile=False)

	current_batch = 0

	for i in range(len(val_data)):

		pred = model.predict_on_batch(val_data[i])

		for batch_id in range(current_batch * batch_size, current_batch * batch_size + pred.shape[0]):

			plt.clf()
			fig = plt.figure()
			ax = fig.add_subplot(111, projection='3d')

			print(batch_id - current_batch * batch_size)

			x_true = [i[0] for i in val_labels[current_batch][batch_id - current_batch * batch_size]]
			y_true = [i[1] for i in val_labels[current_batch][batch_id - current_batch * batch_size]]
			z_true = [i[2] for i in val_labels[current_batch][batch_id - current_batch * batch_size]]

			x_f = [i[0] for i in val_data[current_batch][0][batch_id - current_batch * batch_size]]
			y_f = [i[1] for i in val_data[current_batch][0][batch_id - current_batch * batch_size]]
			z_f = [i[2] for i in val_data[current_batch][0][batch_id - current_batch * batch_size]]

			x_m = [i[0] for i in val_data[current_batch][1][batch_id - current_batch * batch_size]]
			y_m = [i[1] for i in val_data[current_batch][1][batch_id - current_batch * batch_size]]
			z_m = [i[2] for i in val_data[current_batch][1][batch_id - current_batch * batch_size]]

			x_pred = [i[0] for i in pred[batch_id - current_batch * batch_size]]
			y_pred = [i[1] for i in pred[batch_id - current_batch * batch_size]]
			z_pred = [i[2] for i in pred[batch_id - current_batch * batch_size]]
			
			ax.scatter(x_true, y_true, z_true, c='y', marker='.')
			ax.scatter(x_pred, y_pred, z_pred, c='g', marker='.')
			
			ax.scatter(x_f, y_f, z_f, c='r', marker='*', alpha=0.05)
			ax.scatter(x_m, y_m, z_m, c='b', marker='*', alpha=0.05)

			ax.set_xlim([-1, 1])
			ax.set_ylim([-1, 1])
			ax.set_zlim([-1, 1])

			plt.show()
			plt.savefig('./prediction_results-' + fname + '/' + str(batch_id + 1).zfill(4) + '_reg.png', dpi=100)
			plt.close()

		current_batch += 1
	
predict_mn40('model-best', 45, 1, True)
