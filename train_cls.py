from callbacks import Prediction_Plotter, PlotLosses
from data_loader import DataGenerator
from keras.optimizers import SGD, Adam
from losses import sorted_mse_loss, chamfer_loss, gmm_nll_loss
from model import ConditionalTransformerNet
from mpl_toolkits.mplot3d import Axes3D
import h5py
import keras
import matplotlib
import matplotlib.pyplot as plt
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[2]
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from keras import backend as K
import numpy as np

matplotlib.use('AGG')

def plot_results(attribute, output, filename):
	plt.figure(figsize=(16,10))
	for name, history in output:
		val = plt.plot(history.epoch, history.history['val_' + attribute], '--', label=name.title()+' Val')
		plt.plot(history.epoch, history.history[attribute], color=val[0].get_color(), label=name.title()+' Train')
	plt.xlabel('Epochs')
	plt.ylabel(attribute)
	plt.legend()
	plt.xlim([0, max(history.epoch)])
	plt.yscale('log')
	plt.savefig(filename + '.png', dpi=250)
	plt.close()

def main():
	train_file = './ModelNet40/ply_data_train.h5'
	test_file = './ModelNet40/ply_data_test.h5'

	num_epochs = 500
	batch_size = 3 * 64

	loss_name = str(sys.argv[1])
	if loss_name == 'sorted_mse_loss': loss = sorted_mse_loss
	if loss_name == 'chamfer_loss': loss = chamfer_loss
	if loss_name == 'gmm_nll_loss': loss = gmm_nll_loss

	train = DataGenerator(train_file, batch_size)
	
	train_data = []     # store all the generated data batches
	train_labels = []   # store all the generated ground_truth batches
	max_iter = 1        # maximum number of iterations, in each iteration one batch is generated; the proper value depends on batch size and size of whole data
	i = 0
	for d, l in train.generator():
		train_data.append(d)
		train_labels.append(l)
		i += 1
		if i == max_iter:
			break
	fixed_len = train_data[0][0].shape[1]
	moving_len = train_data[0][1].shape[1]

	assert fixed_len == moving_len, 'Lengths not consistent'
	num_points = fixed_len

	first_train_X = train_data[0]
	first_train_Y = train_labels[0]
	Prediction_Plot_Train = Prediction_Plotter(first_train_X, first_train_Y, loss_name + '-train', debug=True)

	val = DataGenerator(test_file, batch_size)

	val_data = []     # store all the generated data batches
	val_labels = []   # store all the generated ground_truth batches
	max_iter = 1      # maximum number of iterations, in each iteration one batch is generated; the proper value depends on batch size and size of whole data
	i = 0
	for d, l in val.generator():
		val_data.append(d)
		val_labels.append(l)
		i += 1
		if i == max_iter:
			break

	first_val_X = val_data[0]
	first_val_Y = val_labels[0]
	Prediction_Plot_Val = Prediction_Plotter(first_val_X, first_val_Y, loss_name + '-val', debug=True)

	LossPlotter = PlotLosses()

	'''
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	x1 = [i[0] for i in val_data[0][0][0]] # Fixed X (Red)
	y1 = [i[1] for i in val_data[0][0][0]] # Fixed Y (Red)
	z1 = [i[2] for i in val_data[0][0][0]] # Fixed Z (Red)

	x2 = [i[0] for i in val_data[0][1][0]] # Moved X (Blue)
	y2 = [i[1] for i in val_data[0][1][0]] # Moved Y (Blue)
	z2 = [i[2] for i in val_data[0][1][0]] # Moved Z (Blue)

	x4 = [i[0] for i in val_labels[0][0]]  # Ground Truth X (Yellow)
	y4 = [i[1] for i in val_labels[0][0]]  # Ground Truth Y (Yellow)
	z4 = [i[2] for i in val_labels[0][0]]  # Ground Truth Z (Yellow)

	ax.scatter(x1, y1, z1, c='r', marker='.')
	ax.scatter(x2, y2, z2, c='b', marker='.')
	ax.scatter(x4, y4, z4, c='y', marker='.')
	
	ax.set_xlim([-1, 1])
	ax.set_ylim([-1, 1])
	ax.set_zlim([-1, 1])

	plt.show()
	plt.savefig('pre_reg-scatter.png', dpi=250)
	plt.close()

	y_true = tf.convert_to_tensor(first_val_Y[0], np.float32)
	probs_true = tf.fill([K.int_shape(y_true)[0]], 1 / K.int_shape(y_true)[0])
	covariance_matrix = tfp.stats.covariance(y_true)
	covariance_matrix = np.diag([0.00001, 0.00001, 0.00001])
	covariance_matrix = tf.constant(covariance_matrix, dtype=tf.float32)

	mix_gauss_true = tfd.MixtureSameFamily(
		mixture_distribution=tfd.Categorical(
			probs=probs_true),
		components_distribution=tfd.MultivariateNormalFullCovariance(
			loc=y_true,
			covariance_matrix=covariance_matrix))

	x = np.linspace(-1, 1, 75, dtype=np.float32)
	y = np.linspace(-1, 1, 75, dtype=np.float32)
	z = np.linspace(-1, 1, 75, dtype=np.float32)

	A = np.vstack(np.meshgrid(x,y,z)).reshape(3,-1).T
	x1 = [i for i in A.T[0]]
	y1 = [i for i in A.T[1]]
	z1 = [i for i in A.T[2]]

	cval = K.eval(mix_gauss_true.prob(A))
	for i in range(len(cval)-1, -1, -1):
		if cval[i] <= 1e-5:
			cval = np.delete(cval, i, 0)
			x1.pop(i)
			y1.pop(i)
			z1.pop(i)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(x1, y1, z1, c=cval, marker='.', cmap='plasma')
	ax.set_xlim([-1, 1])
	ax.set_ylim([-1, 1])
	ax.set_zlim([-1, 1])

	plt.show()
	plt.savefig('test.png', dpi=1000)

	return
	'''
	
	model = ConditionalTransformerNet(num_points)
	learning_rate = 0.05
	opt = Adam(lr=learning_rate)
	model.compile(optimizer=opt,
				  loss=loss)

	if not os.path.exists('./results/'):
		os.mkdir('./results/')

	f = h5py.File(train_file, mode='r')
	num_train = f['data'].shape[0]
	f = h5py.File(test_file, mode='r')
	num_val = f['data'].shape[0]

	history = model.fit_generator(train.generator(),
								  steps_per_epoch=num_train // batch_size,
								  epochs=num_epochs,
								  validation_data=val.generator(),
								  validation_steps=num_val // batch_size,
								  callbacks=[Prediction_Plot_Train, Prediction_Plot_Val, LossPlotter],
								  verbose=1)
	model.save('./results/CTN-' + loss_name + '.h5')
	name = ''
	output = [(name, history)]
	plot_results('loss', output, loss_name + '-loss') 

if __name__ == '__main__':

	main()
