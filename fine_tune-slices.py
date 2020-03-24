import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf
import wandb
from callbacks import Prediction_Plotter
from predict import get_mr_us_data
from datetime import datetime
from losses import chamfer_loss
from mpl_toolkits.mplot3d import Axes3D
from wandb.keras import WandbCallback
from model import ConditionalTransformerNet, TPSTransformNet, MatMul
matplotlib.use('AGG')
os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[1]

try:
	v = int(tf.VERSION[0])
except AttributeError:
	v = int(tf.__version__[0])

if v >= 2:
	from tensorflow.keras.models import load_model
	from tensorflow.keras.layers import Layer
	from tensorflow.keras.callbacks import ModelCheckpoint
	from tensorflow.keras.optimizers import Adam
else:
	from keras.models import load_model
	from keras.engine.topology import Layer
	from keras.callbacks import ModelCheckpoint
	from tensorflow.keras.optimizers import Adam

def fine_tune(learning_rate, freeze):
	if not os.path.exists('./results' + str(sys.argv[1]) + '/'):
		os.mkdir('./results' + str(sys.argv[1]) + '/')

	batch_size = 32
	loss_func = chamfer_loss
	loss_name = 'chamfer_loss'
	dims = [2048, 3]

	#RUN_ID = '2onkrc2a'
	wandb.init(project="ctn-chamfer-fine_tune", name='TPSSlices + lr' + str(learning_rate) + ' freeze' + str(freeze), reinit=True)#, resume=RUN_ID)

	if not os.path.exists('./mrus/prostates.npy') or not os.path.exists('./mrus/prostate_metrics.npy'):
		all_prostates, _ = get_mr_us_data('./mrus/us_labels_resampled800_post3.h5', './mrus/mr_labels_resampled800_post3.h5')
		np.save('./mrus/prostates.npy', all_prostates)
		np.save('./mrus/prostate_metrics.npy', metrics)
	else:
		all_prostates = np.load('./mrus/prostates.npy', allow_pickle=True)
		metrics = np.load('./mrus/prostate_metrics.npy', allow_pickle=True)

	max_iters = len(all_prostates)
	X1 = []
	X2 = []
	X3 = []
	Y = []
	for i in range(0, max_iters // 2):

		fixed_prostate = all_prostates[i][0][0]
		fixed_prostate_X = fixed_prostate[fixed_prostate[:, 0] <= 0.02, :]
		fixed_prostate_X = fixed_prostate_X[fixed_prostate_X[:, 0] >= -0.02, :]
		fixed_prostate_Y = fixed_prostate[fixed_prostate[:, 1] <= 0.02, :]
		fixed_prostate_Y = fixed_prostate_Y[fixed_prostate_Y[:, 1] >= -0.02, :]
		fixed_prostate_slices = np.concatenate((fixed_prostate_X, fixed_prostate_Y), axis=0)
		if fixed_prostate.shape[0] > dims[0]:
			fixed_prostate = fixed_prostate[np.random.choice(fixed_prostate.shape[0], size=dims[0], replace=False), :]
		else:
			fixed_prostate = np.resize(fixed_prostate, dims)
		if fixed_prostate_slices.shape[0] > dims[0]:
			fixed_prostate_slices = fixed_prostate_slices[np.random.choice(fixed_prostate_slices.shape[0], size=dims[0], replace=False), :]
		else:
			fixed_prostate_slices = np.resize(fixed_prostate_slices, dims)
		
		moving_prostate = all_prostates[i][0][1]
		moving_prostate = moving_prostate[np.random.choice(moving_prostate.shape[0], size=dims[0], replace=False), :]

		# Make each data the Fixed, Moving, Moved
		X1.append(np.array(fixed_prostate_slices))
		X2.append(np.array(moving_prostate))
		X3.append(np.array(moving_prostate))
		Y.append(np.array(fixed_prostate))
	X_train = np.array([np.array(X1), np.array(X2), np.array(X3)])
	Y_train = np.array(Y)

	X1 = []
	X2 = []
	X3 = []
	Y = []
	for i in range(max_iters // 2, max_iters):

		fixed_prostate = all_prostates[i][0][0]
		fixed_prostate_X = fixed_prostate[fixed_prostate[:, 0] <= 0.02, :]
		fixed_prostate_X = fixed_prostate_X[fixed_prostate_X[:, 0] >= -0.02, :]
		fixed_prostate_Y = fixed_prostate[fixed_prostate[:, 1] <= 0.02, :]
		fixed_prostate_Y = fixed_prostate_Y[fixed_prostate_Y[:, 1] >= -0.02, :]
		fixed_prostate_slices = np.concatenate((fixed_prostate_X, fixed_prostate_Y), axis=0)
		if fixed_prostate.shape[0] > dims[0]:
			fixed_prostate = fixed_prostate[np.random.choice(fixed_prostate.shape[0], size=dims[0], replace=False), :]
		else:
			fixed_prostate = np.resize(fixed_prostate, dims)
		if fixed_prostate_slices.shape[0] > dims[0]:
			fixed_prostate_slices = fixed_prostate_slices[np.random.choice(fixed_prostate_slices.shape[0], size=dims[0], replace=False), :]
		else:
			fixed_prostate_slices = np.resize(fixed_prostate_slices, dims)
		
		moving_prostate = all_prostates[i][0][1]
		moving_prostate = moving_prostate[np.random.choice(moving_prostate.shape[0], size=dims[0], replace=False), :]

		# Make each data the Fixed, Moving, Moved
		X1.append(np.array(fixed_prostate_slices))
		X2.append(np.array(moving_prostate))
		X3.append(np.array(moving_prostate))
		Y.append(np.array(fixed_prostate))
	X_test = np.array([np.array(X1), np.array(X2), np.array(X3)])
	Y_test = np.array(Y)

	Prediction_Plot_Trn = Prediction_Plotter([X for X in X_train],
											 Y_train, 
											 './results' + str(sys.argv[1]) + '/' + loss_name + '-train')

	Prediction_Plot_Val = Prediction_Plotter([X for X in X_test],
											 Y_test, 
											 './results' + str(sys.argv[1]) + '/' + loss_name + '-val')

	#model = load_model('no-fine-tune_BASE.h5', custom_objects={'MatMul':MatMul, 'chamfer_loss':chamfer_loss})
	init_epoch = 0 #wandb.run.step
	model = TPSTransformNet(2048)
	model.load_weights('tps27.h5')#wandb.restore('model-best.h5').name)

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
						epochs=10000,
						initial_epoch=init_epoch,
						validation_data=([X for X in X_test], Y_test),
						callbacks=[Prediction_Plot_Val, 
								   Prediction_Plot_Trn,
								   WandbCallback()],
						verbose=2)

	model.save('./results' + str(sys.argv[1]) + '/CTN-' + loss_name + '.h5')
	model.save(os.path.join(wandb.run.dir, "model.h5"))

fine_tune(learning_rate=float(sys.argv[2]), freeze=0)
