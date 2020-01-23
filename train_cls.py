import h5py
import keras
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf
import tensorflow_probability as tfp
from callbacks import Prediction_Plotter, PlotLosses
from data_loader import DataGenerator
from datetime import datetime
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import SGD, Adam
from losses import chamfer_loss, gmm_nll_loss
from model import ConditionalTransformerNet, TPSTransformNet
from mpl_toolkits.mplot3d import Axes3D
os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[2]
tfd = tfp.distributions

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

	num_epochs = 1000
	batch_size = 3 * 32

	loss_name = str(sys.argv[1])
	loss_func = None
	metrics = None
	
	if loss_name == 'chamfer_loss':
		loss_func = chamfer_loss
		learning_rate = 1e-2
	
	elif loss_name == 'gmm_nll_loss':
		loss_func = gmm_nll_loss
		metrics = [chamfer_loss]
		learning_rate = 1e-4
	
	else:
		loss_func = gmm_nll_loss

	train = DataGenerator(train_file, batch_size, deform=True)
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
	num_points = fixed_len

	first_train_X = train_data[0]
	first_train_Y = train_labels[0]
	Prediction_Plot_Train = Prediction_Plotter(first_train_X, first_train_Y, loss_name + '-train')

	val = DataGenerator(test_file, batch_size, deform=True)
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
	Prediction_Plot_Val = Prediction_Plotter(first_val_X, first_val_Y, loss_name + '-val')

	LossPlotter = PlotLosses()

	logdir = "./logs/CTN_" + datetime.now().strftime("%Y%m%d-%H%M%S")
	checkpointer = ModelCheckpoint(filepath='./logs/CTN_Model_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.hdf5', verbose=0, save_best_only=True)
	lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-10)

	model = ConditionalTransformerNet(num_points)
	opt = Adam(lr=learning_rate)
	model.compile(optimizer=opt,
				  loss=loss_func,
				  metrics=metrics)

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
								  callbacks=[Prediction_Plot_Train, Prediction_Plot_Val, LossPlotter, checkpointer, lr_reducer],
								  verbose=2)
	model.save('./results/CTN-' + loss_name + '.h5')
	name = ''
	output = [(name, history)]
	plot_results('loss', output, loss_name + '-loss') 

if __name__ == '__main__':

	main()
