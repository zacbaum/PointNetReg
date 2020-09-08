import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf
import wandb
from callbacks import Prediction_Plotter
from predict_prostate import get_mr_us_data
from losses import chamfer_loss, chamfer_loss_batch, gmm_nll_loss
from mpl_toolkits.mplot3d import Axes3D
from wandb.keras import WandbCallback
from model import FreePointTransformer, TPSTransformNet, MatMul

matplotlib.use("AGG")
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

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


def fine_tune(batch_size, learning_rate, freeze, train_method="full"):
	if not os.path.exists("./results" + str(sys.argv[1]) + "/"):
		os.mkdir("./results" + str(sys.argv[1]) + "/")

	loss_name = str(sys.argv[2])
	loss_func = None
	metric = []

	if loss_name == "chamfer_loss":
		loss_func = chamfer_loss

	if loss_name == "chamfer_loss_batch":
		loss_func = chamfer_loss_batch

	if loss_name == "gmm_nll_loss":
		loss_func = gmm_nll_loss(0.001, 0.1)
		metric = [chamfer_loss_batch]

	if not os.path.exists("./mrus/prostates.npy") or not os.path.exists(
		"./mrus/prostate_metrics.npy"
	):
		all_prostates, _ = get_mr_us_data(
			"./mrus/us_labels_resampled800_post3.h5",
			"./mrus/mr_labels_resampled800_post3.h5",
		)
		np.save("./mrus/prostates.npy", all_prostates)
		np.save("./mrus/prostate_metrics.npy", metrics)
	else:
		all_prostates = np.load("./mrus/prostates.npy", allow_pickle=True)
		metrics = np.load("./mrus/prostate_metrics.npy", allow_pickle=True)

	max_iters = len(all_prostates)
	split = 0.7
	dims = [2048, 3]

	X1 = []
	X2 = []
	X3 = []
	Y = []
	for i in range(0, int(split * max_iters)):

		if train_method == "full":
			fixed_prostate = all_prostates[i][0][0]
			fixed_prostate = fixed_prostate[
				np.random.choice(fixed_prostate.shape[0], size=dims[0], replace=False),
				:,
			]

			moving_prostate = all_prostates[i][0][1]
			moving_prostate = moving_prostate[
				np.random.choice(moving_prostate.shape[0], size=dims[0], replace=False),
				:,
			]

		if train_method == "slices":
			fixed_prostate = all_prostates[i][0][0]
			fixed_prostate_X = fixed_prostate[fixed_prostate[:, 0] <= 0.02, :]
			fixed_prostate_X = fixed_prostate_X[fixed_prostate_X[:, 0] >= -0.02, :]
			fixed_prostate_Y = fixed_prostate[fixed_prostate[:, 1] <= 0.02, :]
			fixed_prostate_Y = fixed_prostate_Y[fixed_prostate_Y[:, 1] >= -0.02, :]
			fixed_prostate_slices = np.concatenate(
				(fixed_prostate_X, fixed_prostate_Y), axis=0
			)
			if fixed_prostate.shape[0] > dims[0]:
				fixed_prostate = fixed_prostate[
					np.random.choice(
						fixed_prostate.shape[0], size=dims[0], replace=False
					),
					:,
				]
			else:
				fixed_prostate = np.resize(fixed_prostate, dims)
			if fixed_prostate_slices.shape[0] > dims[0]:
				fixed_prostate_slices = fixed_prostate_slices[
					np.random.choice(
						fixed_prostate_slices.shape[0], size=dims[0], replace=False
					),
					:,
				]
			else:
				fixed_prostate_slices = np.resize(fixed_prostate_slices, dims)

			moving_prostate = all_prostates[i][0][1]
			moving_prostate = moving_prostate[
				np.random.choice(moving_prostate.shape[0], size=dims[0], replace=False),
				:,
			]

		if train_method == "landmarks":
			fixed_prostate = all_prostates[i][0][0]
			ROIs = [x for x in all_prostates[i][1:]]
			if ROIs != []:
				fixed_ROIs = [x[0] for x in ROIs]
				fixed_ROIs_u = np.array([np.unique(x[0], axis=0) for x in ROIs])
				num_points = sum(ROI.shape[0] for ROI in fixed_ROIs_u)
				fixed_prostate = fixed_prostate[
					np.random.choice(
						fixed_prostate.shape[0],
						size=fixed_prostate.shape[0] - num_points,
						replace=False,
					),
					:,
				]
				for ROI in fixed_ROIs_u:
					fixed_prostate = np.vstack((fixed_prostate, ROI))

			moving_prostate = all_prostates[i][0][1]
			ROIs = [x for x in all_prostates[i][1:]]
			if ROIs != []:
				moving_ROIs = [x[1] for x in ROIs]
				moving_ROIs_u = [np.unique(x[1], axis=0) for x in ROIs]
				num_points = sum(ROI.shape[0] for ROI in moving_ROIs_u)
				moving_prostate = moving_prostate[
					np.random.choice(
						moving_prostate.shape[0],
						size=moving_prostate.shape[0] - num_points,
						replace=False,
					),
					:,
				]
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
	for i in range(int(split * max_iters), max_iters):

		if train_method == "full":
			fixed_prostate = all_prostates[i][0][0]
			fixed_prostate = fixed_prostate[
				np.random.choice(fixed_prostate.shape[0], size=dims[0], replace=False),
				:,
			]

			moving_prostate = all_prostates[i][0][1]
			moving_prostate = moving_prostate[
				np.random.choice(moving_prostate.shape[0], size=dims[0], replace=False),
				:,
			]

		if train_method == "slices":
			fixed_prostate = all_prostates[i][0][0]
			fixed_prostate_X = fixed_prostate[fixed_prostate[:, 0] <= 0.02, :]
			fixed_prostate_X = fixed_prostate_X[fixed_prostate_X[:, 0] >= -0.02, :]
			fixed_prostate_Y = fixed_prostate[fixed_prostate[:, 1] <= 0.02, :]
			fixed_prostate_Y = fixed_prostate_Y[fixed_prostate_Y[:, 1] >= -0.02, :]
			fixed_prostate_slices = np.concatenate(
				(fixed_prostate_X, fixed_prostate_Y), axis=0
			)
			if fixed_prostate.shape[0] > dims[0]:
				fixed_prostate = fixed_prostate[
					np.random.choice(
						fixed_prostate.shape[0], size=dims[0], replace=False
					),
					:,
				]
			else:
				fixed_prostate = np.resize(fixed_prostate, dims)
			if fixed_prostate_slices.shape[0] > dims[0]:
				fixed_prostate_slices = fixed_prostate_slices[
					np.random.choice(
						fixed_prostate_slices.shape[0], size=dims[0], replace=False
					),
					:,
				]
			else:
				fixed_prostate_slices = np.resize(fixed_prostate_slices, dims)

			moving_prostate = all_prostates[i][0][1]
			moving_prostate = moving_prostate[
				np.random.choice(moving_prostate.shape[0], size=dims[0], replace=False),
				:,
			]

		if train_method == "landmarks":
			fixed_prostate = all_prostates[i][0][0]
			ROIs = [x for x in all_prostates[i][1:]]
			if ROIs != []:
				fixed_ROIs = [x[0] for x in ROIs]
				fixed_ROIs_u = np.array([np.unique(x[0], axis=0) for x in ROIs])
				num_points = sum(ROI.shape[0] for ROI in fixed_ROIs_u)
				fixed_prostate = fixed_prostate[
					np.random.choice(
						fixed_prostate.shape[0],
						size=fixed_prostate.shape[0] - num_points,
						replace=False,
					),
					:,
				]
				for ROI in fixed_ROIs_u:
					fixed_prostate = np.vstack((fixed_prostate, ROI))

			moving_prostate = all_prostates[i][0][1]
			ROIs = [x for x in all_prostates[i][1:]]
			if ROIs != []:
				moving_ROIs = [x[1] for x in ROIs]
				moving_ROIs_u = [np.unique(x[1], axis=0) for x in ROIs]
				num_points = sum(ROI.shape[0] for ROI in moving_ROIs_u)
				moving_prostate = moving_prostate[
					np.random.choice(
						moving_prostate.shape[0],
						size=moving_prostate.shape[0] - num_points,
						replace=False,
					),
					:,
				]
				for ROI in moving_ROIs_u:
					moving_prostate = np.vstack((moving_prostate, ROI))

		# Make each data the Fixed, Moving, Moved
		X1.append(np.array(fixed_prostate))
		X2.append(np.array(moving_prostate))
		X3.append(np.array(moving_prostate))
		Y.append(np.array(fixed_prostate))
	X_test = np.array([np.array(X1), np.array(X2), np.array(X3)])
	Y_test = np.array(Y)

	Prediction_Plot_Trn = Prediction_Plotter(
		[X for X in X_train],
		Y_train,
		"./results" + str(sys.argv[1]) + "/" + loss_name + "-train",
	)

	Prediction_Plot_Val = Prediction_Plotter(
		[X for X in X_test],
		Y_test,
		"./results" + str(sys.argv[1]) + "/" + loss_name + "-val",
	)
	'''
	wandb.init(
		project="fpt_journal",
		name="PN4D MRUS bs" + str(batch_size) + " lr" + str(learning_rate),
		#reinit=True,
		#resume=RUN_ID,
	)
	'''
	model = FreePointTransformer(
		dims[0],
		dims=3,
		pn_filters=[64, 128, 1024],
		ctn_filters=[1024, 512, 256, 128, 64],
		skips=False,
	)
	'''
	model = TPSTransformNet(
		dims[0],
		dims=3,
		tps_features=27,
		sigma=1.0,
	)
	'''

	init_epoch = 0
	'''
	model = load_model(
		wandb.restore("model-best.h5").name,
		custom_objects={"MatMul": MatMul, loss_name: loss_func},
	)
	initial_epoch = wandb.run.step
	'''

	if freeze == 1:  # Freeze everything except output
		trainable_layers = ["conv1d_17"]
		for layer in model.layers:
			for name in trainable_layers:
				if name in layer.name:
					break
				else:
					layer.trainable = False
			else:
				continue
			break

	if freeze == 2:  # Freeze only the pointnet
		trainable_layers = [
			"conv1d_12",
			"conv1d_13",
			"conv1d_14",
			"conv1d_15",
			"conv1d_16",
			"conv1d_17",
		]
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
	model.compile(optimizer=optimizer, loss=loss_func, metrics=metric)

	history = model.fit(
		[X for X in X_train],
		Y_train,
		batch_size,
		epochs=100,
		initial_epoch=init_epoch,
		validation_data=([X for X in X_test], Y_test),
		callbacks=[
			Prediction_Plot_Trn, 
			Prediction_Plot_Val, 
			#WandbCallback()
		],
		verbose=1,
	)

	model.save("./results" + str(sys.argv[1]) + "/CTN-" + loss_name + ".h5")
	model.save(os.path.join(wandb.run.dir, "model.h5"))

if __name__ == "__main__":

	learning_rate = float(sys.argv[3])
	batch_size = int(sys.argv[4])

	fine_tune(batch_size, learning_rate, freeze=0)
