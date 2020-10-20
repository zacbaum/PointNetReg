import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf
import wandb
from callbacks import Prediction_Plotter
from tensorflow.keras.callbacks import ModelCheckpoint
from data_loader_prostate import DataGenerator
from predict_prostate import get_mr_us_data
from losses import chamfer_loss, chamfer_loss_batch, gmm_nll_loss
from mpl_toolkits.mplot3d import Axes3D
from wandb.keras import WandbCallback
from model import FreePointTransformer, TPSTransformNet, MatMul

matplotlib.use("AGG")
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]

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


def fine_tune(
	batch_size, 
	learning_rate, 
	network_type,
	rotate=0,
	displace=0,
	deform=0,
	shuffle=True,
	shuffle_points=True,
	kept_points=2048,
	slices=False,
	sweeps=0,
	train_split=0.7,
	epochs=5000,
):

	test_split = 0.7

	#results_path = "./results-PN4D-" + network_type + "-gmm-MRUS-" + str(train_split) + "-sweeps" + str(sweeps) + "-d" + str(displace) + "-r" + str(rotate) + "-bs" + str(batch_size) + "-lr" + str(learning_rate) + "/"
	results_path = "./results-PN4D-" + str(kept_points) + "-" + network_type + "-MRUS-" + str(train_split) + "-d" + str(displace) + "-r" + str(rotate) + "-bs" + str(batch_size) + "-lr" + str(learning_rate) + "/"
	if not os.path.exists(results_path):
		os.mkdir(results_path)

	loss_name = str(sys.argv[1])
	loss_func = None
	metric = []

	if loss_name == "chamfer_loss":
		loss_func = chamfer_loss

	if loss_name == "chamfer_loss_batch":
		loss_func = chamfer_loss_batch

	if loss_name == "gmm_nll_loss":
		loss_func = gmm_nll_loss(0.001, 0.1)
		metric = [chamfer_loss_batch]

	if not os.path.exists("./mrus/prostates_volumes.npy") or not os.path.exists("./mrus/prostate_metrics_volumes.npy"):
		all_prostates, _ = get_mr_us_data(
			"./mrus/us_labels_resampled800_post3.h5",
			"./mrus/mr_labels_resampled800_post3.h5",
		)
		np.save("./mrus/prostates_volumes.npy", all_prostates)
		np.save("./mrus/prostate_metrics_volumes.npy", metrics)
	else:
		all_prostates = np.load("./mrus/prostates_volumes.npy", allow_pickle=True)
		metrics = np.load("./mrus/prostate_metrics_volumes.npy", allow_pickle=True)

	max_iters = len(all_prostates)

	X1 = []
	X2 = []
	for i in range(0, int(train_split * max_iters)):
			
		fixed_prostate = all_prostates[i][0][0]
		moving_prostate = all_prostates[i][0][1]

		# Make each data the Fixed, Moving
		X1.append(np.array(fixed_prostate))
		X2.append(np.array(moving_prostate))

	X_train = np.array([np.array(X1), np.array(X2)])

	X1 = []
	X2 = []
	for i in range(int(test_split * max_iters), max_iters):

		fixed_prostate = all_prostates[i][0][0]
		moving_prostate = all_prostates[i][0][1]

		# Make each data the Fixed, Moving
		X1.append(np.array(fixed_prostate))
		X2.append(np.array(moving_prostate))

	X_test = np.array([np.array(X1), np.array(X2)])

	train = DataGenerator(
		X_train,
		batch_size,
		dims=4,
		shuffle=shuffle,
		shuffle_points=shuffle_points,
		rotate=rotate,
		displace=displace,
		deform=deform,
		kept_points=kept_points,
		slices=slices,
		sweeps=sweeps,
	)

	test = DataGenerator(
		X_test,
		batch_size,
		dims=4,
		shuffle=shuffle,
		shuffle_points=shuffle_points,
		rotate=0,
		displace=0,
		deform=0,
		kept_points=kept_points,
		slices=slices,
		sweeps=sweeps,
	)

	trn_data = []  # store all the generated data batches
	trn_labels = []  # store all the generated ground_truth batches
	max_iter = 1  # maximum number of iterations, in each iteration one batch is generated; the proper value depends on batch size and size of whole data
	i = 0
	for d, l in train:
		trn_data.append(d)
		trn_labels.append(l)
		i += 1
		if i == max_iter:
			break
	
	first_trn_X = trn_data[0]
	first_trn_Y = trn_labels[0]
	Prediction_Plot_Trn = Prediction_Plotter(
		first_trn_X,
		first_trn_Y,
		results_path + loss_name + "-trn",
	)

	val_data = []  # store all the generated data batches
	val_labels = []  # store all the generated ground_truth batches
	max_iter = 1  # maximum number of iterations, in each iteration one batch is generated; the proper value depends on batch size and size of whole data
	i = 0
	for d, l in test:
		val_data.append(d)
		val_labels.append(l)
		i += 1
		if i == max_iter:
			break
	
	first_val_X = val_data[0]
	first_val_Y = val_labels[0]
	Prediction_Plot_Val = Prediction_Plotter(
		first_val_X,
		first_val_Y,
		results_path + loss_name + "-val",
	)

	wandb.init(
		project="fpt-journal-volumes",
		#name="PN4D-" + network_type + " MRUS " + str(train_split) + " gmm sweeps" + str(sweeps) + " w/ d" + str(displace) + " r" + str(rotate) + " bs" + str(batch_size) + " lr" + str(learning_rate),
		name="PN4D-" + str(kept_points) + "-" + network_type + " MRUS " + str(train_split) + " w/ d" + str(displace) + " r" + str(rotate) + " bs" + str(batch_size) + " lr" + str(learning_rate),
		reinit=True,
		#resume='gkti67bl',
	)

	if network_type == 'fpt':
		model = FreePointTransformer(
			kept_points,
			dims=4,
			skips=False,
		)
	if network_type == 'tps':
		model = TPSTransformNet(
			kept_points,
			dims=4,
			tps_features=27,
			sigma=1.0,
		)

	init_epoch = 0
	'''
	model = load_model(
		wandb.restore("model-best.h5").name,
		custom_objects={"MatMul": MatMul, loss_name: loss_func},
	)
	initial_epoch = wandb.run.step
	'''
	optimizer = Adam(lr=learning_rate)
	model.compile(optimizer=optimizer, loss=loss_func, metrics=metric)

	history = model.fit_generator(
		train,
		steps_per_epoch=int(train_split * max_iters) // batch_size,
		epochs=epochs,
		initial_epoch=init_epoch,
		validation_data=test,
		callbacks=[
			Prediction_Plot_Trn,
			Prediction_Plot_Val, 
			WandbCallback(log_weights=True),
		],
		verbose=2,
	)

	model.save(os.path.join(wandb.run.dir, "model.h5"))

if __name__ == "__main__":

	learning_rate = float(sys.argv[3])
	batch_size = int(sys.argv[4])

	# batch size, learning rate, network_type, rotate, displace, deform, shuffle, shuff_pts, keep, slices, sweeps, train_split, epochs

	fine_tune(batch_size, learning_rate, 'fpt', 45, 1, 0, True, True, 2**13, False, 0, 0.7, 2000)
	fine_tune(batch_size, learning_rate, 'tps', 45, 1, 0, True, True, 2**13, False, 0, 0.7, 2000)

	fine_tune(batch_size, learning_rate, 'fpt', 45, 1, 0, True, True, 2**12, False, 0, 0.7, 2000)
	fine_tune(batch_size, learning_rate, 'tps', 45, 1, 0, True, True, 2**12, False, 0, 0.7, 2000)

	fine_tune(batch_size, learning_rate, 'fpt', 45, 1, 0, True, True, 2**11, False, 0, 0.7, 2000)
	fine_tune(batch_size, learning_rate, 'tps', 45, 1, 0, True, True, 2**11, False, 0, 0.7, 2000)

	'''
	fine_tune(batch_size, learning_rate, 'fpt', 45, 1, 0, True, True, 2048, False, 3, 0.7, 2000)
	fine_tune(batch_size, learning_rate, 'fpt', 45, 1, 0, True, True, 2048, False, 4, 0.7, 2000)
	fine_tune(batch_size, learning_rate, 'fpt', 45, 1, 0, True, True, 2048, False, 5, 0.7, 2000)
	fine_tune(batch_size, learning_rate, 'fpt', 45, 1, 0, True, True, 2048, False, 6, 0.7, 2000)

	fine_tune(batch_size, learning_rate, 'tps', 45, 1, 0, True, True, 2048, False, 3, 0.7, 2000)
	fine_tune(batch_size, learning_rate, 'tps', 45, 1, 0, True, True, 2048, False, 4, 0.7, 2000)
	fine_tune(batch_size, learning_rate, 'tps', 45, 1, 0, True, True, 2048, False, 5, 0.7, 2000)
	fine_tune(batch_size, learning_rate, 'tps', 45, 1, 0, True, True, 2048, False, 6, 0.7, 2000)

	fine_tune(batch_size, learning_rate, 'fpt', 45, 1, 0, True, True, 2048, False, 3, 0.6, 2000)
	fine_tune(batch_size, learning_rate, 'fpt', 45, 1, 0, True, True, 2048, False, 4, 0.6, 2000)
	fine_tune(batch_size, learning_rate, 'fpt', 45, 1, 0, True, True, 2048, False, 5, 0.6, 2000)
	fine_tune(batch_size, learning_rate, 'fpt', 45, 1, 0, True, True, 2048, False, 6, 0.6, 2000)

	fine_tune(batch_size, learning_rate, 'tps', 45, 1, 0, True, True, 2048, False, 3, 0.6, 2000)
	fine_tune(batch_size, learning_rate, 'tps', 45, 1, 0, True, True, 2048, False, 4, 0.6, 2000)
	fine_tune(batch_size, learning_rate, 'tps', 45, 1, 0, True, True, 2048, False, 5, 0.6, 2000)
	fine_tune(batch_size, learning_rate, 'tps', 45, 1, 0, True, True, 2048, False, 6, 0.6, 2000)

	fine_tune(batch_size, learning_rate, 'fpt', 45, 1, 0, True, True, 2048, False, 3, 0.5, 2000)
	fine_tune(batch_size, learning_rate, 'fpt', 45, 1, 0, True, True, 2048, False, 4, 0.5, 2000)
	fine_tune(batch_size, learning_rate, 'fpt', 45, 1, 0, True, True, 2048, False, 5, 0.5, 2000)
	fine_tune(batch_size, learning_rate, 'fpt', 45, 1, 0, True, True, 2048, False, 6, 0.5, 2000)

	fine_tune(batch_size, learning_rate, 'tps', 45, 1, 0, True, True, 2048, False, 3, 0.5, 2000)
	fine_tune(batch_size, learning_rate, 'tps', 45, 1, 0, True, True, 2048, False, 4, 0.5, 2000)
	fine_tune(batch_size, learning_rate, 'tps', 45, 1, 0, True, True, 2048, False, 5, 0.5, 2000)
	fine_tune(batch_size, learning_rate, 'tps', 45, 1, 0, True, True, 2048, False, 6, 0.5, 2000)

	fine_tune(batch_size, learning_rate, 'fpt', 45, 1, 0, True, True, 2048, False, 3, 0.4, 2000)
	fine_tune(batch_size, learning_rate, 'fpt', 45, 1, 0, True, True, 2048, False, 4, 0.4, 2000)
	fine_tune(batch_size, learning_rate, 'fpt', 45, 1, 0, True, True, 2048, False, 5, 0.4, 2000)
	fine_tune(batch_size, learning_rate, 'fpt', 45, 1, 0, True, True, 2048, False, 6, 0.4, 2000)

	fine_tune(batch_size, learning_rate, 'tps', 45, 1, 0, True, True, 2048, False, 3, 0.4, 2000)
	fine_tune(batch_size, learning_rate, 'tps', 45, 1, 0, True, True, 2048, False, 4, 0.4, 2000)
	fine_tune(batch_size, learning_rate, 'tps', 45, 1, 0, True, True, 2048, False, 5, 0.4, 2000)
	fine_tune(batch_size, learning_rate, 'tps', 45, 1, 0, True, True, 2048, False, 6, 0.4, 2000)

	fine_tune(batch_size, learning_rate, 'fpt', 45, 1, 0, True, True, 2048, False, 3, 0.3, 2000)
	fine_tune(batch_size, learning_rate, 'fpt', 45, 1, 0, True, True, 2048, False, 4, 0.3, 2000)
	fine_tune(batch_size, learning_rate, 'fpt', 45, 1, 0, True, True, 2048, False, 5, 0.3, 2000)
	fine_tune(batch_size, learning_rate, 'fpt', 45, 1, 0, True, True, 2048, False, 6, 0.3, 2000)
	
	fine_tune(batch_size, learning_rate, 'tps', 45, 1, 0, True, True, 2048, False, 3, 0.3, 2000)
	fine_tune(batch_size, learning_rate, 'tps', 45, 1, 0, True, True, 2048, False, 4, 0.3, 2000)
	fine_tune(batch_size, learning_rate, 'tps', 45, 1, 0, True, True, 2048, False, 5, 0.3, 2000)
	fine_tune(batch_size, learning_rate, 'tps', 45, 1, 0, True, True, 2048, False, 6, 0.3, 2000)
	'''
