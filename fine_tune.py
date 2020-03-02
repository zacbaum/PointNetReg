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
from model import ConditionalTransformerNet, TPSTransformNet
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
	from keras.optimizers import Adam

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

def fine_tune(learning_rate, freeze):
	if not os.path.exists('./results' + str(sys.argv[1]) + '/'):
		os.mkdir('./results' + str(sys.argv[1]) + '/')
	if not os.path.exists('./logs' + str(sys.argv[1]) + '/'):
		os.mkdir('./logs' + str(sys.argv[1]) + '/')

	batch_size = 32
	loss_func = chamfer_loss
	loss_name = 'chamfer_loss'

	#wandb.init(project="ctn-chamfer-fine_tune", name='500ROI + lr' + str(learning_rate) + ' freeze' + str(freeze), reinit=True)

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
		ROIs = [x for x in all_prostates[i][1:]]
		if ROIs != []:
			fixed_ROIs = [x[0] for x in ROIs]
			fixed_ROIs_u = np.array([np.unique(x[0], axis=0) for x in ROIs])
			num_points = sum(ROI.shape[0] for ROI in fixed_ROIs_u)
			fixed_prostate = fixed_prostate[np.random.choice(fixed_prostate.shape[0], size=fixed_prostate.shape[0] - num_points, replace=False), :]
			for ROI in fixed_ROIs_u:
				fixed_prostate = np.vstack((fixed_prostate, ROI))
		
		moving_prostate = all_prostates[i][0][1]
		ROIs = [x for x in all_prostates[i][1:]]
		if ROIs != []:
			moving_ROIs = [x[1] for x in ROIs]
			moving_ROIs_u = [np.unique(x[1], axis=0) for x in ROIs]
			num_points = sum(ROI.shape[0] for ROI in moving_ROIs_u)
			moving_prostate = moving_prostate[np.random.choice(moving_prostate.shape[0], size=moving_prostate.shape[0] - num_points, replace=False), :]
			for ROI in moving_ROIs_u:
				moving_prostate = np.vstack((moving_prostate, ROI))

		# Make each data the Fixed, Moving, Moved
		X1.append(np.array(fixed_prostate))
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
		ROIs = [x for x in all_prostates[i][1:]]
		if ROIs != []:
			fixed_ROIs = [x[0] for x in ROIs]
			fixed_ROIs_u = np.array([np.unique(x[0], axis=0) for x in ROIs])
			num_points = sum(ROI.shape[0] for ROI in fixed_ROIs_u)
			fixed_prostate = fixed_prostate[np.random.choice(fixed_prostate.shape[0], size=fixed_prostate.shape[0] - num_points, replace=False), :]
			for ROI in fixed_ROIs_u:
				fixed_prostate = np.vstack((fixed_prostate, ROI))
		
		moving_prostate = all_prostates[i][0][1]
		ROIs = [x for x in all_prostates[i][1:]]
		if ROIs != []:
			moving_ROIs = [x[1] for x in ROIs]
			moving_ROIs_u = [np.unique(x[1], axis=0) for x in ROIs]
			num_points = sum(ROI.shape[0] for ROI in moving_ROIs_u)
			moving_prostate = moving_prostate[np.random.choice(moving_prostate.shape[0], size=moving_prostate.shape[0] - num_points, replace=False), :]
			for ROI in moving_ROIs_u:
				moving_prostate = np.vstack((moving_prostate, ROI))

		# Make each data the Fixed, Moving, Moved
		X1.append(np.array(fixed_prostate))
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

	logdir = "./logs" + str(sys.argv[1]) + "/CTN_" + datetime.now().strftime("%Y%m%d-%H%M%S")
	checkpointer = ModelCheckpoint(filepath='./logs' + str(sys.argv[1]) + '/CTN_Model_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.h5',
								   verbose=0,
								   save_best_only=True)

	model = load_model('no-fine-tune_BASE.h5', custom_objects={'MatMul':MatMul, 'chamfer_loss':chamfer_loss})
	init_epoch = 0
	#model = ConditionalTransformerNet(2048, dropout=0.0, batch_norm=False)

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
						initial_epoch=init_epoch,
						validation_data=([X for X in X_test], Y_test),
						callbacks=[Prediction_Plot_Val, 
								   Prediction_Plot_Trn,
								   checkpointer,
								   WandbCallback()],
						verbose=2)

	model.save('./results' + str(sys.argv[1]) + '/CTN-' + loss_name + '.h5')
	model.save(os.path.join(wandb.run.dir, "model.h5"))

fine_tune(learning_rate=float(sys.argv[2]), freeze=0)
fine_tune(learning_rate=float(sys.argv[2]), freeze=1)
fine_tune(learning_rate=float(sys.argv[2]), freeze=2)
