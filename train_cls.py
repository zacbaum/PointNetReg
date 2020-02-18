import h5py
import keras
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf
import wandb
from callbacks import Prediction_Plotter
from data_loader import DataGenerator
from datetime import datetime
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from losses import chamfer_loss, variational_loss
from model import ConditionalTransformerNet
from mpl_toolkits.mplot3d import Axes3D
from wandb.keras import WandbCallback
matplotlib.use('AGG')
os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[2]

#from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction  = 0.48
#set_session(tf.Session(config=config))

def main():
	train_file = './ModelNet40/ply_data_train.h5'
	test_file = './ModelNet40/ply_data_test.h5'
	if not os.path.exists('./results' + str(sys.argv[2]) + '/'):
		os.mkdir('./results' + str(sys.argv[2]) + '/')
	if not os.path.exists('./logs' + str(sys.argv[2]) + '/'):
		os.mkdir('./logs' + str(sys.argv[2]) + '/')

	batch_size = 32
	scale = 1
	load_from_file = False

	loss_name = str(sys.argv[1])
	loss_func = None
	learning_rate = float(sys.argv[3])

	wandb.init(project="ctn-chamfer", name='d0.0 bn0 lr1e-3')

	if loss_name == 'chamfer_loss':
		loss_func = chamfer_loss

	if loss_name == 'variational_loss':
		loss_func = variational_loss

	train = DataGenerator(train_file,
						  batch_size,
						  scale=scale,
						  deform=True,
						  part=0)

	val = DataGenerator(test_file,
						batch_size,
						scale=scale,
						deform=True,
						part=0)
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
	Prediction_Plot_Val = Prediction_Plotter(first_val_X,
											 first_val_Y, 
											 './results' + str(sys.argv[2]) + '/' + loss_name + '-val',
											 scale)
	fixed_len = val_data[0][0].shape[1]
	moving_len = val_data[0][1].shape[1]
	assert (fixed_len == moving_len)
	num_points = fixed_len

	logdir = "./logs" + str(sys.argv[2]) + "/CTN_" + datetime.now().strftime("%Y%m%d-%H%M%S")
	checkpointer = ModelCheckpoint(filepath='./logs' + str(sys.argv[2]) + '/CTN_Model_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.h5',
								   verbose=0,
								   save_best_only=True)

	if load_from_file:
		from keras.models import load_model
		from keras.engine.topology import Layer
		from model import MatMul
		model = load_model('CTN-chamfer_loss.h5', custom_objects={'MatMul':MatMul, 'chamfer_loss':chamfer_loss})
		initial_epoch = 1000
	else:
		model = ConditionalTransformerNet(num_points, dropout=0.0, batch_norm=True)
		optimizer = Adam(lr=learning_rate)
		model.compile(optimizer=optimizer,
					  loss=loss_func)
		initial_epoch = 0
	
	f = h5py.File(train_file, mode='r')
	num_train = f['data'].shape[0]
	f = h5py.File(test_file, mode='r')
	num_val = f['data'].shape[0]

	history = model.fit_generator(train.generator(),
								  steps_per_epoch=num_train // batch_size,
								  epochs=5000,
								  initial_epoch=initial_epoch,
								  validation_data=val.generator(),
								  validation_steps=num_val // batch_size,
								  callbacks=[Prediction_Plot_Val, 
											 checkpointer,
											 WandbCallback()],
								  verbose=1)

	model.save('./results' + str(sys.argv[2]) + '/CTN-' + loss_name + '.h5')
	model.save(os.path.join(wandb.run.dir, "model.h5"))

if __name__ == '__main__':

	main()
