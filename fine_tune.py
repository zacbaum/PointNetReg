import h5py
import keras
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf
import wandb
import scipy.io as sio
from sklearn.model_selection import train_test_split
from callbacks import Prediction_Plotter
from predict import get_filenames, get_indices, get_prostate_data
from data_loader import DataGenerator
from datetime import datetime
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from losses import chamfer_loss
from mpl_toolkits.mplot3d import Axes3D
from wandb.keras import WandbCallback
matplotlib.use('AGG')
os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[1]

def fine_tune(learning_rate, freeze):
	if not os.path.exists('./results' + str(sys.argv[1]) + '/'):
		os.mkdir('./results' + str(sys.argv[1]) + '/')
	if not os.path.exists('./logs' + str(sys.argv[1]) + '/'):
		os.mkdir('./logs' + str(sys.argv[1]) + '/')

	batch_size = 32
	loss_func = chamfer_loss
	loss_name = 'chamfer_loss'

	wandb.init(project="ctn-chamfer-fine_tune", name='lr' + str(learning_rate) + ' freeze' + str(freeze))

	prostate_data = sio.loadmat('prostate.mat')
	filenames = get_filenames(prostate_data)
	indxs  = get_indices(prostate_data, filenames)
	all_prostates = get_prostate_data(prostate_data, indxs)

	perms = [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]]
	max_iters = len(all_prostates) - 2
	X1 = []
	X2 = []
	X3 = []
	Y = []
	for i in range(0, max_iters // 3, 3):
		for perm in perms:
			# Make each data the Fixed, Moving, Moved
			X1.append(np.array(all_prostates[i + perm[0]][0]))
			X2.append(np.array(all_prostates[i + perm[1]][0]))
			X3.append(np.array(all_prostates[i + perm[1]][0]))
			Y.append(np.array(all_prostates[i + perm[0]][0]))
	X_train = np.array([np.array(X1), np.array(X2), np.array(X3)])
	Y_train = np.array(Y)

	X1 = []
	X2 = []
	X3 = []
	Y = []
	for i in range(max_iters // 3, max_iters, 3):
		for perm in perms:
			# Make each data the Fixed, Moving, Moved
			X1.append(all_prostates[i + perm[0]][0])
			X2.append(all_prostates[i + perm[1]][0])
			X3.append(all_prostates[i + perm[1]][0])
			Y.append(all_prostates[i + perm[0]][0])
	X_test = np.array([np.array(X1), np.array(X2), np.array(X3)])
	Y_test = np.array(Y)

	Prediction_Plot_Trn = Prediction_Plotter([X for X in X_train],
											 Y_train, 
											 './results' + str(sys.argv[1]) + '/' + loss_name + '-train',
											 1)

	Prediction_Plot_Val = Prediction_Plotter([X for X in X_test],
											 Y_test, 
											 './results' + str(sys.argv[1]) + '/' + loss_name + '-val',
											 1)

	logdir = "./logs" + str(sys.argv[1]) + "/CTN_" + datetime.now().strftime("%Y%m%d-%H%M%S")
	checkpointer = ModelCheckpoint(filepath='./logs' + str(sys.argv[1]) + '/CTN_Model_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.h5',
								   verbose=0,
								   save_best_only=True)

	from keras.models import load_model
	from keras.engine.topology import Layer
	from model import MatMul
	model = load_model('chamfer-lr1e-3-2000.h5', custom_objects={'MatMul':MatMul, 'chamfer_loss':chamfer_loss})

	if freeze == 1: # Freeze everything except output
		trainable_layers = ['conv1d_17']
		for layer in model.layers:
			for name in trainable_layers:
				if name in layer.name:
					break
				else:
					layer.trainable = False
			else:
				continue
			break

	if freeze == 2: # Freeze only the pointnet
		trainable_layers = ['conv1d_12', 'conv1d_13', 'conv1d_14', 'conv1d_15', 'conv1d_16', 'conv1d_17']
		for layer in model.layers:
			for name in trainable_layers:
				if name in layer.name:
					break
				else:
					layer.trainable = False
			else:
				continue
			break

	for layer in model.layers:
		print(layer.name, layer.trainable)

	optimizer = Adam(lr=learning_rate)
	model.compile(optimizer=optimizer,
				  loss=loss_func)

	history = model.fit([X for X in X_train],
						Y_train,
						batch_size,
						epochs=100,
						validation_data=([X for X in X_test], Y_test),
						callbacks=[Prediction_Plot_Val, 
								   Prediction_Plot_Trn,
								   checkpointer,
								   WandbCallback()],
						verbose=2)

	model.save('./results' + str(sys.argv[1]) + '/CTN-' + loss_name + '.h5')
	model.save(os.path.join(wandb.run.dir, "model.h5"))

fine_tune(learning_rate=float(sys.argv[2]), freeze=2)