import tensorflow as tf
from keras import backend as K
import numpy as np
import scipy
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

def kabsch(a, b):
    assert a.shape == b.shape
    n, dim = a.shape

    cent_a = a - a.mean(axis=0)
    cent_b = b - b.mean(axis=0)

    H = np.dot(np.transpose(cent_a), cent_b) / n

    V, S, W = np.linalg.svd(H)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    R = np.dot(V, W)
    t = b.mean(axis=0) - a.mean(axis=0).dot(R)

    return R, t

def predict_mn40(fname, rotate, displace, deform, part, part_nn, noise):
	if not os.path.exists('./prediction_results-' + fname + '/'):
			os.mkdir('./prediction_results-' + fname + '/')

	header_string = 'P_ID\tDC_P\tDH_P\tR_deg\tt\n'
	f = open('./prediction_results-' + fname + '/results_log.txt', 'a')
	f.write(header_string)
	f.close()

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
						part=part, part_nn=int(2048 * part_nn),
						noise=noise)

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
			
			#ax.scatter(x_f, y_f, z_f, c='r', marker='*', alpha=0.05)
			#ax.scatter(x_m, y_m, z_m, c='b', marker='*', alpha=0.05)

			ax.set_xlim([-1, 1])
			ax.set_ylim([-1, 1])
			ax.set_zlim([-1, 1])

			plt.show()
			plt.savefig('./prediction_results-' + fname + '/' + str(batch_id + 1).zfill(4) + '_reg.png', dpi=100)
			plt.close()

			a_points = val_labels[current_batch][batch_id - current_batch * batch_size]
			b_points = pred[batch_id - current_batch * batch_size]
			if a_points.shape[1] == 4:
				a_points = a_points[:, :-1]
			if b_points.shape[1] == 4:
				b_points = b_points[:, :-1]
			
			cd = round(chamfer(a_points, b_points), 5)
			hd = round(hausdorff_2way(a_points, b_points), 5)
			
			R, t = kabsch(a_points, b_points)
			deg_R = round(np.degrees(np.arccos((np.trace(R) - 1) / 2)), 5)
			mag_t = round(np.linalg.norm(t), 5)

			result_string = str(batch_id + 1) + '\t' \
						  + str(cd) + '\t' \
						  + str(hd) + '\t' \
						  + str(deg_R) + '\t' \
						  + str(mag_t) + '\n'
			f = open('./prediction_results-' + fname + '/results_log.txt', 'a')
			f.write(result_string)
			f.close()

		current_batch += 1

predict_mn40('PN-4D-3154-00', rotate=00, displace=1, deform=False, part=0, part_nn=0.75, noise=0)
predict_mn40('PN-4D-3154-10', rotate=10, displace=1, deform=False, part=0, part_nn=0.75, noise=0)
predict_mn40('PN-4D-3154-20', rotate=20, displace=1, deform=False, part=0, part_nn=0.75, noise=0)
predict_mn40('PN-4D-3154-30', rotate=30, displace=1, deform=False, part=0, part_nn=0.75, noise=0)
predict_mn40('PN-4D-3154-40', rotate=40, displace=1, deform=False, part=0, part_nn=0.75, noise=0)
predict_mn40('PN-4D-3154-50', rotate=50, displace=1, deform=False, part=0, part_nn=0.75, noise=0)
predict_mn40('PN-4D-3154-60', rotate=60, displace=1, deform=False, part=0, part_nn=0.75, noise=0)
predict_mn40('PN-4D-3154-70', rotate=70, displace=1, deform=False, part=0, part_nn=0.75, noise=0)
predict_mn40('PN-4D-3154-80', rotate=80, displace=1, deform=False, part=0, part_nn=0.75, noise=0)
predict_mn40('PN-4D-3154-90', rotate=90, displace=1, deform=False, part=0, part_nn=0.75, noise=0)

predict_mn40('PN-4D-3154-00d', rotate=00, displace=1, deform=True, part=0, part_nn=0.75, noise=0)
predict_mn40('PN-4D-3154-10d', rotate=10, displace=1, deform=True, part=0, part_nn=0.75, noise=0)
predict_mn40('PN-4D-3154-20d', rotate=20, displace=1, deform=True, part=0, part_nn=0.75, noise=0)
predict_mn40('PN-4D-3154-30d', rotate=30, displace=1, deform=True, part=0, part_nn=0.75, noise=0)
predict_mn40('PN-4D-3154-40d', rotate=40, displace=1, deform=True, part=0, part_nn=0.75, noise=0)
predict_mn40('PN-4D-3154-50d', rotate=50, displace=1, deform=True, part=0, part_nn=0.75, noise=0)
predict_mn40('PN-4D-3154-60d', rotate=60, displace=1, deform=True, part=0, part_nn=0.75, noise=0)
predict_mn40('PN-4D-3154-70d', rotate=70, displace=1, deform=True, part=0, part_nn=0.75, noise=0)
predict_mn40('PN-4D-3154-80d', rotate=80, displace=1, deform=True, part=0, part_nn=0.75, noise=0)
predict_mn40('PN-4D-3154-90d', rotate=90, displace=1, deform=True, part=0, part_nn=0.75, noise=0)
