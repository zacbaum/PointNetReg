import tensorflow as tf
from keras import backend as K
import numpy as np
import scipy
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('AGG')
import os
import re
os.environ["CUDA_VISIBLE_DEVICES"]='CPU'
import time
from cpd import deformable_registration, gaussian_kernel
from model import MatMul, FreePointTransformer
import h5py
import cv2

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

def get_unique_plot_points(points):
	points_u = np.unique(points, axis=0)
	x = [i[0] for i in points_u]
	y = [i[1] for i in points_u]
	z = [i[2] for i in points_u]
	return [x, y, z]

def get_mr_us_data(fixed_fname, moving_fname, dims=[2048,3]):
	
	fixed = h5py.File(fixed_fname, mode='r')
	fixed_keys = list(fixed.keys())
	moving = h5py.File(moving_fname, mode='r')
	moving_keys = list(moving.keys())
	num_labels = fixed['num_labels'][0]
	pxl_to_mm_scale = 0.8
	
	all_prostates = []
	all_prostates_metrics = []
	indx = 0
	for case in range(len(num_labels)):
		current = []
		current_prostate_metrics = []
		for mask in range(num_labels[case]):
			
			fixed_points = np.array(fixed[fixed_keys[indx]])
			fixed_contour_points = []
			for z in range(fixed_points.shape[0]):
				edged = cv2.Canny(fixed_points[z], 0, 1)
				indices = np.where(edged != [0])
				coordinates = zip(indices[0], indices[1])
				for coordinate in coordinates:
					fixed_contour_points.append([pxl_to_mm_scale * coordinate[0], pxl_to_mm_scale * coordinate[1], pxl_to_mm_scale * z])
			fixed_contour_points = np.array(fixed_contour_points)
			
			moving_points = np.array(moving[moving_keys[indx]])
			moving_contour_points = []
			for z in range(moving_points.shape[0]):
				edged = cv2.Canny(moving_points[z], 0, 1)
				indices = np.where(edged != [0])
				coordinates = zip(indices[0], indices[1])
				for coordinate in coordinates:
					moving_contour_points.append([pxl_to_mm_scale * coordinate[0], pxl_to_mm_scale * coordinate[1], pxl_to_mm_scale * z])
			moving_contour_points = np.array(moving_contour_points)
			
			if current_prostate_metrics == []:
				fixed_contour_points_mean = np.mean(fixed_contour_points, axis=0)
				fixed_contour_points = fixed_contour_points - fixed_contour_points_mean
				fixed_contour_points_min = np.min(fixed_contour_points)
				fixed_contour_points_ptp = np.ptp(fixed_contour_points)
				fixed_contour_points = 2 * (fixed_contour_points - fixed_contour_points_min) / fixed_contour_points_ptp - 1
				fixed_prostate_metrics = [fixed_contour_points_mean, fixed_contour_points_min, fixed_contour_points_ptp]
				moving_contour_points_mean = np.mean(moving_contour_points, axis=0)
				moving_contour_points = moving_contour_points - moving_contour_points_mean
				moving_contour_points_min = np.min(moving_contour_points)
				moving_contour_points_ptp = np.ptp(moving_contour_points)
				moving_contour_points = 2 * (moving_contour_points - moving_contour_points_min) / moving_contour_points_ptp - 1
				moving_prostate_metrics = [moving_contour_points_mean, moving_contour_points_min, moving_contour_points_ptp]
			else: 
				fixed_contour_points = fixed_contour_points - fixed_prostate_metrics[0]
				fixed_contour_points = 2 * (fixed_contour_points - fixed_prostate_metrics[1]) / fixed_prostate_metrics[2] - 1
				moving_contour_points = moving_contour_points - moving_prostate_metrics[0]
				moving_contour_points = 2 * (moving_contour_points - moving_prostate_metrics[1]) / moving_prostate_metrics[2] - 1

			current.append([fixed_contour_points, moving_contour_points])
			if current_prostate_metrics == []:
				current_prostate_metrics.append([fixed_prostate_metrics, moving_prostate_metrics])
			indx += 1
		
		all_prostates.append(current)
		all_prostates_metrics.append(current_prostate_metrics)
	return all_prostates, all_prostates_metrics

def set_plot_ax_lims(axes, limit=1):
	for ax in axes:
		ax.set_xlim([-limit, limit])
		ax.set_ylim([-limit, limit])
		ax.set_zlim([-limit, limit])
		ax.grid(False)
		ax.set_axis_off()

def denormalize(unique_points, reference_points):
	unique_points_dn = 0.5 * ((unique_points * np.ptp(reference_points)) + (2 * np.min(reference_points)) + np.ptp(reference_points))
	return unique_points_dn

def denormalize_from_metrics(unique_points, minimum, ptp):
	unique_points_dn = 0.5 * ((unique_points * ptp) + (2 * minimum) + ptp)
	return unique_points_dn

def predict_mr_us_file(fname):
	if not os.path.exists('./prostate_results-' + fname + '/'):
			os.mkdir('./prostate_results-' + fname + '/')

	header_string = 'P_ID\tTIME\tDC_P\tDH_P\tDC_R1\tDH_R1\tTRE_R1\tDC_R2\tDH_R2\tTRE_R2\tDC_R3\tDH_R3\tTRE_R3\tDC_R4\tDH_R4\tTRE_R4\tDC_R5\tDH_R5\tTRE_R5\n'
	f = open('./prostate_results-' + fname + '/P2P.txt', 'a')
	f.write(header_string)
	f.close()
	'''
	f = open('./prostate_results-' + fname + '/CPD-P2P.txt', 'a')
	f.write(header_string)
	f.close()
	# Load the model.
	model = TPSTransformNet(2048)
	model.load_weights(fname + '.h5')
	'''

	dims = [2048, 3]
	if not os.path.exists('./mrus/prostates.npy') or not os.path.exists('./mrus/prostate_metrics.npy'):
		all_prostates, metrics = get_mr_us_data('./mrus/us_labels_resampled800_post3.h5', './mrus/mr_labels_resampled800_post3.h5')
		np.save('./mrus/prostates.npy', all_prostates)
		np.save('./mrus/prostate_metrics.npy', metrics)
	else:
		all_prostates = np.load('./mrus/prostates.npy', allow_pickle=True)
		metrics = np.load('./mrus/prostate_metrics.npy', allow_pickle=True)

	max_iters = len(all_prostates)

	# CTN - Contour to Contour
	for i in range(max_iters):

		# Fixed Prostate
		fixed_prostate = all_prostates[i][0][0]
		if fixed_prostate.shape[0] > dims[0]:
			fixed_prostate = fixed_prostate[np.random.choice(fixed_prostate.shape[0], size=dims[0], replace=False), :]
		else:
			fixed_prostate = np.resize(fixed_prostate, dims)
		fixed_prostate_u = np.unique(fixed_prostate, axis=0)
		# Fixed ROIs
		ROIs = [x for x in all_prostates[i][1:]]
		fixed_ROIs = [x[0] for x in ROIs]
		for r in range(len(fixed_ROIs)):
			if fixed_ROIs[r].shape[0] > dims[0]:
				fixed_ROIs[r] = fixed_ROIs[r][np.random.choice(r.shape[0], size=dims[0], replace=False), :]
			else:
				fixed_ROIs[r] = np.resize(fixed_ROIs[r], dims)
		fixed_ROIs_u = [np.unique(x[0], axis=0) for x in ROIs]

		# Moving Prostate
		moving_prostate = all_prostates[i][0][1]
		if moving_prostate.shape[0] > dims[0]:
			moving_prostate = moving_prostate[np.random.choice(moving_prostate.shape[0], size=dims[0], replace=False), :]
		else:
			moving_prostate = np.resize(moving_prostate, dims)
		moving_prostate_u = np.unique(moving_prostate, axis=0)
		# Moving ROIs
		ROIs = [x for x in all_prostates[i][1:]]
		moving_ROIs = [x[1] for x in ROIs]
		for r in range(len(moving_ROIs)):
			if moving_ROIs[r].shape[0] > dims[0]:
				moving_ROIs[r] = moving_ROIs[r][np.random.choice(r.shape[0], size=dims[0], replace=False), :]
			else:
				moving_ROIs[r] = np.resize(moving_ROIs[r], dims)
		moving_ROIs_u = [np.unique(x[1], axis=0) for x in ROIs]

		# Moving2Fixed Prostate
		t = time.time()
		pred = moving_prostate
		#pred = model.predict([[np.array(fixed_prostate)],
		#					  [np.array(moving_prostate)],
		#					  [np.array(moving_prostate)]])
		t = round(time.time() - t, 3)
		pred_u = np.unique(pred, axis=0)
		#pred_u = np.unique(pred[0], axis=0)
		
		# Moving2Fixed ROIs
		pred_ROIs = moving_ROIs
		#pred_ROIs = [model.predict([[np.array(fixed_prostate)],
		#						   	[np.array(moving_prostate)],
		#						   	[np.array(x)]]) for x in moving_ROIs]
		pred_ROIs_u = [np.unique(x, axis=0) for x in pred_ROIs]
		#pred_ROIs_u = [np.unique(x[0], axis=0) for x in pred_ROIs]
		
		# Scale the data so we can compute metrics with correct values.
		fixed_metrics = metrics[i][0][0]
		moving_metrics = metrics[i][0][1]

		fixed_prostate_dn = denormalize_from_metrics(fixed_prostate_u, fixed_metrics[1], fixed_metrics[2])
		moving_prostate_dn = denormalize_from_metrics(moving_prostate_u, moving_metrics[1], moving_metrics[2])
		pred_dn = denormalize_from_metrics(pred_u, fixed_metrics[1], fixed_metrics[2])
		d_c_p = round(chamfer(fixed_prostate_dn, pred_dn), 3)
		d_h_p = round(hausdorff_2way(fixed_prostate_dn, pred_dn), 3)

		fixed_ROIs_dn = [denormalize_from_metrics(x, fixed_metrics[1], fixed_metrics[2]) for x in fixed_ROIs_u]
		pred_ROIs_dn = [denormalize_from_metrics(x, fixed_metrics[1], fixed_metrics[2]) for x in pred_ROIs_u]
		d_c_ROIs = [round(chamfer(fixed_ROIs_dn[x], pred_ROIs_dn[x]), 3) for x in range(len(fixed_ROIs_dn))]
		d_h_ROIs = [round(hausdorff_2way(fixed_ROIs_dn[x], pred_ROIs_dn[x]), 3) for x in range(len(fixed_ROIs_dn))]
		d_RE_ROIs = [round(np.linalg.norm(np.mean(fixed_ROIs_dn[x], axis=0) - np.mean(pred_ROIs_dn[x], axis=0)), 3) for x in range(len(fixed_ROIs_dn))]

		x_fp, y_fp, z_fp = get_unique_plot_points(fixed_prostate_dn)
		fixed_ROIs_xyz = [get_unique_plot_points(x) for x in fixed_ROIs_dn]

		x_mp, y_mp, z_mp = get_unique_plot_points(moving_prostate_dn)
		moving_ROIs_xyz = [get_unique_plot_points(x) for x in moving_ROIs_u]

		x_pred, y_pred, z_pred = get_unique_plot_points(pred_dn)
		pred_ROIs_xyz = [get_unique_plot_points(x) for x in pred_ROIs_dn]

		fig = plt.figure()
		ax0 = fig.add_subplot(221, projection='3d')
		ax0.set_title('Fixed')
		ax0.scatter(x_fp, y_fp, z_fp, c='y', marker='.', alpha=0.2)
		
		ax1 = fig.add_subplot(222, projection='3d')
		ax1.set_title('Moving')
		ax1.scatter(x_mp, y_mp, z_mp, c='r', marker='.', alpha=0.2)
	
		ax2 = fig.add_subplot(223, projection='3d')
		ax2.set_title('Registered Contours')
		#ax2.scatter(x_fp, y_fp, z_fp, c='y', marker='.', alpha=0.2)
		ax2.scatter(x_pred, y_pred, z_pred, c='g', marker='.', alpha=0.2)
	
		ax3 = fig.add_subplot(224, projection='3d')
		ax3.set_title('Registered ROIs')
		for roi in fixed_ROIs_xyz:
			ax3.scatter(roi[0], roi[1], roi[2], c='y', marker='.', alpha=0.1)
		for roi in pred_ROIs_xyz:
			ax3.scatter(roi[0], roi[1], roi[2], c='g', marker='.', alpha=0.1)

		set_plot_ax_lims([ax0, ax1, ax2, ax3], limit=30)

		ROIs_string = ''
		for roi in range(5):
			d_c = str(d_c_ROIs[roi]) if roi < len(d_c_ROIs) else ' '
			d_h = str(d_h_ROIs[roi]) if roi < len(d_h_ROIs) else ' '
			d_RE = str(d_RE_ROIs[roi]) if roi < len(d_RE_ROIs) else ' '
			ROIs_string += str(d_c) + '\t' + str(d_h) + '\t' + str(d_RE) + '\t'
		result_string = str(i + 1) + '\t' + str(t) + '\t' + \
							str(d_c_p) + '\t' + str(d_h_p) + '\t' + \
							ROIs_string + '\n'
		f = open('./prostate_results-' + fname + '/P2P.txt', 'a')
		f.write(result_string)
		f.close()

		fig.suptitle('Patient ' + str(i + 1) + ' - MR to US')
		plt.show()
		plt.savefig('./prostate_results-' + fname + '/P2P-' + str(i + 1) + '.png', dpi=300)
		plt.close()
	'''
	# CPD - Contour to Contour
	for i in range(0, max_iters):

		# Fixed Prostate
		fixed_prostate = all_prostates[i][0][0]
		if fixed_prostate.shape[0] > dims[0]:
			fixed_prostate = fixed_prostate[np.random.choice(fixed_prostate.shape[0], size=dims[0], replace=False), :]
		else:
			fixed_prostate = np.resize(fixed_prostate, dims)
		fixed_prostate_u = np.unique(fixed_prostate, axis=0)
		# Fixed ROIs
		ROIs = [x for x in all_prostates[i][1:]]
		fixed_ROIs = [x[0] for x in ROIs]
		for r in range(len(fixed_ROIs)):
			if fixed_ROIs[r].shape[0] > dims[0]:
				fixed_ROIs[r] = fixed_ROIs[r][np.random.choice(r.shape[0], size=dims[0], replace=False), :]
			else:
				fixed_ROIs[r] = np.resize(fixed_ROIs[r], dims)
		fixed_ROIs_u = [np.unique(x[0], axis=0) for x in ROIs]

		# Moving Prostate
		moving_prostate = all_prostates[i][0][1]
		if moving_prostate.shape[0] > dims[0]:
			moving_prostate = moving_prostate[np.random.choice(moving_prostate.shape[0], size=dims[0], replace=False), :]
		else:
			moving_prostate = np.resize(moving_prostate, dims)
		moving_prostate_u = np.unique(moving_prostate, axis=0)
		# Moving ROIs
		ROIs = [x for x in all_prostates[i][1:]]
		moving_ROIs = [x[1] for x in ROIs]
		for r in range(len(moving_ROIs)):
			if moving_ROIs[r].shape[0] > dims[0]:
				moving_ROIs[r] = moving_ROIs[r][np.random.choice(r.shape[0], size=dims[0], replace=False), :]
			else:
				moving_ROIs[r] = np.resize(moving_ROIs[r], dims)
		moving_ROIs_u = [np.unique(x[1], axis=0) for x in ROIs]

		# Moving2Fixed Prostate
		reg = deformable_registration(**{'X':fixed_prostate, 'Y':moving_prostate, 'max_iterations':150})
		t = time.time()
		pred, params = reg.register()
		t = round(time.time() - t, 3)
		pred_u = np.unique(pred, axis=0)
		# Moving2Fixed ROIs
		pred_ROIs = [x + np.dot(gaussian_kernel(moving_prostate, x), params[1]) for x in moving_ROIs_u]
		pred_ROIs_u = [np.unique(x, axis=0) for x in pred_ROIs]

		# Scale the data so we can compute metrics with correct values.
		fixed_metrics = metrics[i][0][0]
		moving_metrics = metrics[i][0][1]

		fixed_prostate_dn = denormalize_from_metrics(fixed_prostate_u, fixed_metrics[1], fixed_metrics[2])
		moving_prostate_dn = denormalize_from_metrics(moving_prostate_u, moving_metrics[1], moving_metrics[2])
		pred_dn = denormalize_from_metrics(pred_u, fixed_metrics[1], fixed_metrics[2])
		d_c_p = round(chamfer(fixed_prostate_dn, pred_dn), 3)
		d_h_p = round(hausdorff_2way(fixed_prostate_dn, pred_dn), 3)

		fixed_ROIs_dn = [denormalize_from_metrics(x, fixed_metrics[1], fixed_metrics[2]) for x in fixed_ROIs_u]
		pred_ROIs_dn = [denormalize_from_metrics(x, fixed_metrics[1], fixed_metrics[2]) for x in pred_ROIs_u]
		d_c_ROIs = [round(chamfer(fixed_ROIs_dn[x], pred_ROIs_dn[x]), 3) for x in range(len(fixed_ROIs_dn))]
		d_h_ROIs = [round(hausdorff_2way(fixed_ROIs_dn[x], pred_ROIs_dn[x]), 3) for x in range(len(fixed_ROIs_dn))]
		d_RE_ROIs = [round(np.linalg.norm(np.mean(fixed_ROIs_dn[x], axis=0) - np.mean(pred_ROIs_dn[x], axis=0)), 3) for x in range(len(fixed_ROIs_dn))]

		x_fp, y_fp, z_fp = get_unique_plot_points(fixed_prostate_dn)
		fixed_ROIs_xyz = [get_unique_plot_points(x) for x in fixed_ROIs_dn]

		x_mp, y_mp, z_mp = get_unique_plot_points(moving_prostate_dn)
		moving_ROIs_xyz = [get_unique_plot_points(x) for x in moving_ROIs_u]

		x_pred, y_pred, z_pred = get_unique_plot_points(pred_dn)
		pred_ROIs_xyz = [get_unique_plot_points(x) for x in pred_ROIs_dn]

		fig = plt.figure()
		ax0 = fig.add_subplot(221, projection='3d')
		ax0.set_title('Fixed')
		ax0.scatter(x_fp, y_fp, z_fp, c='y', marker='.', alpha=0.2)
		
		ax1 = fig.add_subplot(222, projection='3d')
		ax1.set_title('Moving')
		ax1.scatter(x_mp, y_mp, z_mp, c='r', marker='.', alpha=0.2)
	
		ax2 = fig.add_subplot(223, projection='3d')
		ax2.set_title('Registered Contours')
		#ax2.scatter(x_fp, y_fp, z_fp, c='y', marker='.', alpha=0.2)
		ax2.scatter(x_pred, y_pred, z_pred, c='g', marker='.', alpha=0.2)
	
		ax3 = fig.add_subplot(224, projection='3d')
		ax3.set_title('Registered ROIs')
		for roi in fixed_ROIs_xyz:
			ax3.scatter(roi[0], roi[1], roi[2], c='y', marker='.', alpha=0.1)
		for roi in pred_ROIs_xyz:
			ax3.scatter(roi[0], roi[1], roi[2], c='g', marker='.', alpha=0.1)

		set_plot_ax_lims([ax0, ax1, ax2, ax3], limit=30)

		ROIs_string = ''
		for roi in range(5):
			d_c = str(d_c_ROIs[roi]) if roi < len(d_c_ROIs) else ' '
			d_h = str(d_h_ROIs[roi]) if roi < len(d_h_ROIs) else ' '
			d_RE = str(d_RE_ROIs[roi]) if roi < len(d_RE_ROIs) else ' '
			ROIs_string += str(d_c) + '\t' + str(d_h) + '\t' + str(d_RE) + '\t'
		result_string = str(i + 1) + '\t' + str(t) + '\t' + \
							str(d_c_p) + '\t' + str(d_h_p) + '\t' + \
							ROIs_string + '\n'
		f = open('./prostate_results-' + fname + '/CPD-P2P.txt', 'a')
		f.write(result_string)
		f.close()

		fig.suptitle('Patient ' + str(i + 1) + ' - MR to US')
		plt.show()
		plt.savefig('./prostate_results-' + fname + '/CPD-P2P-' + str(i + 1) + '.png', dpi=300)
		plt.close()
	'''

def predict_mr_us_file_Slices(fname):
	if not os.path.exists('./prostate_results-' + fname + '/'):
			os.mkdir('./prostate_results-' + fname + '/')

	#header_string = 'P_ID\tTIME\tDC_P\tDH_P\tDC_R1\tDH_R1\tTRE_R1\tDC_R2\tDH_R2\tTRE_R2\tDC_R3\tDH_R3\tTRE_R3\tDC_R4\tDH_R4\tTRE_R4\tDC_R5\tDH_R5\tTRE_R5\n'
	#f = open('./prostate_results-' + fname + '/P2P.txt', 'a')
	#f.write(header_string)
	#f.close()
	#f = open('./prostate_results-' + fname + '/CPD-P2P.txt', 'a')
	#f.write(header_string)
	#f.close()
	# Load the model.
	#model = FreePointTransformer(2048)
	#model.load_weights(fname + '.h5')

	dims = [2048, 3]
	if not os.path.exists('./mrus/prostates.npy') or not os.path.exists('./mrus/prostate_metrics.npy'):
		all_prostates, metrics = get_mr_us_data('./mrus/us_labels_resampled800_post3.h5', './mrus/mr_labels_resampled800_post3.h5')
		np.save('./mrus/prostates.npy', all_prostates)
		np.save('./mrus/prostate_metrics.npy', metrics)
	else:
		all_prostates = np.load('./mrus/prostates.npy', allow_pickle=True)
		metrics = np.load('./mrus/prostate_metrics.npy', allow_pickle=True)

	max_iters = len(all_prostates)

	# CTN - Contour to Contour
	for i in range(max_iters):

		# Fixed Prostate
		fixed_prostate = all_prostates[i][0][0]

		fixed_prostate_X = fixed_prostate[fixed_prostate[:, 0] <= 0.02, :]
		fixed_prostate_X = fixed_prostate_X[fixed_prostate_X[:, 0] >= -0.02, :]
		fixed_prostate_Y = fixed_prostate[fixed_prostate[:, 1] <= 0.02, :]
		fixed_prostate_Y = fixed_prostate_Y[fixed_prostate_Y[:, 1] >= -0.02, :]
		fixed_prostate_Y = fixed_prostate_Y[fixed_prostate_Y[:, 0] >= -0.02, :]
		fixed_prostate_slices = np.concatenate((fixed_prostate_X, fixed_prostate_Y), axis=0)

		if fixed_prostate.shape[0] > dims[0]:
			fixed_prostate = fixed_prostate[np.random.choice(fixed_prostate.shape[0], size=dims[0], replace=False), :]
		else:
			fixed_prostate = np.resize(fixed_prostate, dims)
		if fixed_prostate_slices.shape[0] > dims[0]:
			fixed_prostate_slices = fixed_prostate_slices[np.random.choice(fixed_prostate_slices.shape[0], size=dims[0], replace=False), :]
		else:
			fixed_prostate_slices = np.resize(fixed_prostate_slices, dims)

		fixed_prostate_u = np.unique(fixed_prostate, axis=0)
		# Fixed ROIs
		ROIs = [x for x in all_prostates[i][1:]]
		fixed_ROIs = [x[0] for x in ROIs]
		for r in range(len(fixed_ROIs)):
			if fixed_ROIs[r].shape[0] > dims[0]:
				fixed_ROIs[r] = fixed_ROIs[r][np.random.choice(r.shape[0], size=dims[0], replace=False), :]
			else:
				fixed_ROIs[r] = np.resize(fixed_ROIs[r], dims)
		fixed_ROIs_u = [np.unique(x[0], axis=0) for x in ROIs]

		# Moving Prostate
		moving_prostate = all_prostates[i][0][1]
		if moving_prostate.shape[0] > dims[0]:
			moving_prostate = moving_prostate[np.random.choice(moving_prostate.shape[0], size=dims[0], replace=False), :]
		else:
			moving_prostate = np.resize(moving_prostate, dims)
		moving_prostate_u = np.unique(moving_prostate, axis=0)
		# Moving ROIs
		ROIs = [x for x in all_prostates[i][1:]]
		moving_ROIs = [x[1] for x in ROIs]
		for r in range(len(moving_ROIs)):
			if moving_ROIs[r].shape[0] > dims[0]:
				moving_ROIs[r] = moving_ROIs[r][np.random.choice(r.shape[0], size=dims[0], replace=False), :]
			else:
				moving_ROIs[r] = np.resize(moving_ROIs[r], dims)
		moving_ROIs_u = [np.unique(x[1], axis=0) for x in ROIs]

		# Moving2Fixed Prostate
		t = time.time()
		pred = moving_prostate
		#pred = model.predict([[np.array(fixed_prostate_slices)],
		#					  [np.array(moving_prostate)],
		#					  [np.array(moving_prostate)]])
		t = round(time.time() - t, 3)
		pred_u = np.unique(pred, axis=0)
		#pred_u = np.unique(pred[0], axis=0)
		
		# Moving2Fixed ROIs
		pred_ROIs = moving_ROIs
		#pred_ROIs = [model.predict([[np.array(fixed_prostate_slices)],
		#						   	[np.array(moving_prostate)],
		#						   	[np.array(x)]]) for x in moving_ROIs]
		pred_ROIs_u = [np.unique(x, axis=0) for x in pred_ROIs]
		#pred_ROIs_u = [np.unique(x[0], axis=0) for x in pred_ROIs]
		
		# Scale the data so we can compute metrics with correct values.
		fixed_metrics = metrics[i][0][0]
		moving_metrics = metrics[i][0][1]

		fixed_prostate_dn = denormalize_from_metrics(fixed_prostate_u, fixed_metrics[1], fixed_metrics[2])
		moving_prostate_dn = denormalize_from_metrics(moving_prostate_u, moving_metrics[1], moving_metrics[2])
		pred_dn = denormalize_from_metrics(pred_u, fixed_metrics[1], fixed_metrics[2])
		d_c_p = round(chamfer(fixed_prostate_dn, pred_dn), 3)
		d_h_p = round(hausdorff_2way(fixed_prostate_dn, pred_dn), 3)

		fixed_ROIs_dn = [denormalize_from_metrics(x, fixed_metrics[1], fixed_metrics[2]) for x in fixed_ROIs_u]
		pred_ROIs_dn = [denormalize_from_metrics(x, fixed_metrics[1], fixed_metrics[2]) for x in pred_ROIs_u]
		d_c_ROIs = [round(chamfer(fixed_ROIs_dn[x], pred_ROIs_dn[x]), 3) for x in range(len(fixed_ROIs_dn))]
		d_h_ROIs = [round(hausdorff_2way(fixed_ROIs_dn[x], pred_ROIs_dn[x]), 3) for x in range(len(fixed_ROIs_dn))]
		d_RE_ROIs = [round(np.linalg.norm(np.mean(fixed_ROIs_dn[x], axis=0) - np.mean(pred_ROIs_dn[x], axis=0)), 3) for x in range(len(fixed_ROIs_dn))]

		x_fp, y_fp, z_fp = get_unique_plot_points(fixed_prostate_dn)
		fixed_ROIs_xyz = [get_unique_plot_points(x) for x in fixed_ROIs_dn]

		x_mp, y_mp, z_mp = get_unique_plot_points(moving_prostate_dn)
		moving_ROIs_xyz = [get_unique_plot_points(x) for x in moving_ROIs_u]

		x_pred, y_pred, z_pred = get_unique_plot_points(pred_dn)
		pred_ROIs_xyz = [get_unique_plot_points(x) for x in pred_ROIs_dn]

		fig = plt.figure()
		ax0 = fig.add_subplot(221, projection='3d')
		ax0.set_title('Fixed')
		ax0.scatter(x_fp, y_fp, z_fp, c='y', marker='.', alpha=0.2)
		
		ax1 = fig.add_subplot(222, projection='3d')
		ax1.set_title('Moving')
		ax1.scatter(x_mp, y_mp, z_mp, c='r', marker='.', alpha=0.2)
	
		ax2 = fig.add_subplot(223, projection='3d')
		ax2.set_title('Registered Contours')
		#ax2.scatter(x_fp, y_fp, z_fp, c='y', marker='.', alpha=0.2)
		ax2.scatter(x_pred, y_pred, z_pred, c='g', marker='.', alpha=0.2)
	
		ax3 = fig.add_subplot(224, projection='3d')
		ax3.set_title('Registered ROIs')
		for roi in fixed_ROIs_xyz:
			ax3.scatter(roi[0], roi[1], roi[2], c='y', marker='.', alpha=0.1)
		for roi in pred_ROIs_xyz:
			ax3.scatter(roi[0], roi[1], roi[2], c='g', marker='.', alpha=0.1)

		set_plot_ax_lims([ax0, ax1, ax2, ax3], limit=30)

		'''
		ROIs_string = ''
		for roi in range(5):
			d_c = str(d_c_ROIs[roi]) if roi < len(d_c_ROIs) else ' '
			d_h = str(d_h_ROIs[roi]) if roi < len(d_h_ROIs) else ' '
			d_RE = str(d_RE_ROIs[roi]) if roi < len(d_RE_ROIs) else ' '
			ROIs_string += str(d_c) + '\t' + str(d_h) + '\t' + str(d_RE) + '\t'
		result_string = str(i + 1) + '\t' + str(t) + '\t' + \
							str(d_c_p) + '\t' + str(d_h_p) + '\t' + \
							ROIs_string + '\n'
		f = open('./prostate_results-' + fname + '/P2P.txt', 'a')
		f.write(result_string)
		f.close()
		'''

		fig.suptitle('Patient ' + str(i + 1) + ' - MR to US')
		plt.show()
		plt.savefig('./prostate_results-' + fname + '/P2P-' + str(i + 1) + '.png', dpi=300)
		plt.close()
	
	'''
	# CPD - Contour to Contour
	for i in range(0, max_iters):

		# Fixed Prostate
		fixed_prostate = all_prostates[i][0][0]

		centerX = np.random.uniform(-0.33, 0.33)
		fixed_prostate_X = fixed_prostate[fixed_prostate[:, 0] <= (centerX + 0.02), :]
		fixed_prostate_X = fixed_prostate_X[fixed_prostate_X[:, 0] >= (centerX + -0.02), :]
		centerY = np.random.uniform(-0.33, 0.33)
		fixed_prostate_Y = fixed_prostate[fixed_prostate[:, 1] <= (centerY + 0.02), :]
		fixed_prostate_Y = fixed_prostate_Y[fixed_prostate_Y[:, 1] >= (centerY + -0.02), :]
		fixed_prostate_Y = fixed_prostate_Y[fixed_prostate_Y[:, 0] >= (centerX + -0.02), :]
		fixed_prostate_slices = np.concatenate((fixed_prostate_X, fixed_prostate_Y), axis=0)

		if fixed_prostate.shape[0] > dims[0]:
			fixed_prostate = fixed_prostate[np.random.choice(fixed_prostate.shape[0], size=dims[0], replace=False), :]
		else:
			fixed_prostate = np.resize(fixed_prostate, dims)
		if fixed_prostate_slices.shape[0] > dims[0]:
			fixed_prostate_slices = fixed_prostate_slices[np.random.choice(fixed_prostate_slices.shape[0], size=dims[0], replace=False), :]
		else:
			fixed_prostate_slices = np.resize(fixed_prostate_slices, dims)

		fixed_prostate_u = np.unique(fixed_prostate, axis=0)
		# Fixed ROIs
		ROIs = [x for x in all_prostates[i][1:]]
		fixed_ROIs = [x[0] for x in ROIs]
		for r in range(len(fixed_ROIs)):
			if fixed_ROIs[r].shape[0] > dims[0]:
				fixed_ROIs[r] = fixed_ROIs[r][np.random.choice(r.shape[0], size=dims[0], replace=False), :]
			else:
				fixed_ROIs[r] = np.resize(fixed_ROIs[r], dims)
		fixed_ROIs_u = [np.unique(x[0], axis=0) for x in ROIs]

		# Moving Prostate
		moving_prostate = all_prostates[i][0][1]
		if moving_prostate.shape[0] > dims[0]:
			moving_prostate = moving_prostate[np.random.choice(moving_prostate.shape[0], size=dims[0], replace=False), :]
		else:
			moving_prostate = np.resize(moving_prostate, dims)
		moving_prostate_u = np.unique(moving_prostate, axis=0)
		# Moving ROIs
		ROIs = [x for x in all_prostates[i][1:]]
		moving_ROIs = [x[1] for x in ROIs]
		for r in range(len(moving_ROIs)):
			if moving_ROIs[r].shape[0] > dims[0]:
				moving_ROIs[r] = moving_ROIs[r][np.random.choice(r.shape[0], size=dims[0], replace=False), :]
			else:
				moving_ROIs[r] = np.resize(moving_ROIs[r], dims)
		moving_ROIs_u = [np.unique(x[1], axis=0) for x in ROIs]

		# Moving2Fixed Prostate
		reg = deformable_registration(**{'X':fixed_prostate_slices, 'Y':moving_prostate, 'max_iterations':150})
		t = time.time()
		pred, params = reg.register()
		t = round(time.time() - t, 3)
		pred_u = np.unique(pred, axis=0)
		# Moving2Fixed ROIs
		pred_ROIs = [x + np.dot(gaussian_kernel(moving_prostate, x), params[1]) for x in moving_ROIs_u]
		pred_ROIs_u = [np.unique(x, axis=0) for x in pred_ROIs]

		# Scale the data so we can compute metrics with correct values.
		fixed_metrics = metrics[i][0][0]
		moving_metrics = metrics[i][0][1]

		fixed_prostate_dn = denormalize_from_metrics(fixed_prostate_u, fixed_metrics[1], fixed_metrics[2])
		moving_prostate_dn = denormalize_from_metrics(moving_prostate_u, moving_metrics[1], moving_metrics[2])
		pred_dn = denormalize_from_metrics(pred_u, fixed_metrics[1], fixed_metrics[2])
		d_c_p = round(chamfer(fixed_prostate_dn, pred_dn), 3)
		d_h_p = round(hausdorff_2way(fixed_prostate_dn, pred_dn), 3)

		fixed_ROIs_dn = [denormalize_from_metrics(x, fixed_metrics[1], fixed_metrics[2]) for x in fixed_ROIs_u]
		pred_ROIs_dn = [denormalize_from_metrics(x, fixed_metrics[1], fixed_metrics[2]) for x in pred_ROIs_u]
		d_c_ROIs = [round(chamfer(fixed_ROIs_dn[x], pred_ROIs_dn[x]), 3) for x in range(len(fixed_ROIs_dn))]
		d_h_ROIs = [round(hausdorff_2way(fixed_ROIs_dn[x], pred_ROIs_dn[x]), 3) for x in range(len(fixed_ROIs_dn))]
		d_RE_ROIs = [round(np.linalg.norm(np.mean(fixed_ROIs_dn[x], axis=0) - np.mean(pred_ROIs_dn[x], axis=0)), 3) for x in range(len(fixed_ROIs_dn))]

		x_fp, y_fp, z_fp = get_unique_plot_points(fixed_prostate_dn)
		fixed_ROIs_xyz = [get_unique_plot_points(x) for x in fixed_ROIs_dn]

		x_mp, y_mp, z_mp = get_unique_plot_points(moving_prostate_dn)
		moving_ROIs_xyz = [get_unique_plot_points(x) for x in moving_ROIs_u]

		x_pred, y_pred, z_pred = get_unique_plot_points(pred_dn)
		pred_ROIs_xyz = [get_unique_plot_points(x) for x in pred_ROIs_dn]

		fig = plt.figure()
		ax0 = fig.add_subplot(221, projection='3d')
		ax0.set_title('Fixed')
		ax0.scatter(x_fp, y_fp, z_fp, c='y', marker='.', alpha=0.2)
		
		ax1 = fig.add_subplot(222, projection='3d')
		ax1.set_title('Moving')
		ax1.scatter(x_mp, y_mp, z_mp, c='r', marker='.', alpha=0.2)
	
		ax2 = fig.add_subplot(223, projection='3d')
		ax2.set_title('Registered Contours')
		#ax2.scatter(x_fp, y_fp, z_fp, c='y', marker='.', alpha=0.2)
		ax2.scatter(x_pred, y_pred, z_pred, c='g', marker='.', alpha=0.2)
	
		ax3 = fig.add_subplot(224, projection='3d')
		ax3.set_title('Registered ROIs')
		for roi in fixed_ROIs_xyz:
			ax3.scatter(roi[0], roi[1], roi[2], c='y', marker='.', alpha=0.1)
		for roi in pred_ROIs_xyz:
			ax3.scatter(roi[0], roi[1], roi[2], c='g', marker='.', alpha=0.1)

		set_plot_ax_lims([ax0, ax1, ax2, ax3], limit=30)

		ROIs_string = ''
		for roi in range(5):
			d_c = str(d_c_ROIs[roi]) if roi < len(d_c_ROIs) else ' '
			d_h = str(d_h_ROIs[roi]) if roi < len(d_h_ROIs) else ' '
			d_RE = str(d_RE_ROIs[roi]) if roi < len(d_RE_ROIs) else ' '
			ROIs_string += str(d_c) + '\t' + str(d_h) + '\t' + str(d_RE) + '\t'
		result_string = str(i + 1) + '\t' + str(t) + '\t' + \
							str(d_c_p) + '\t' + str(d_h_p) + '\t' + \
							ROIs_string + '\n'
		f = open('./prostate_results-' + fname + '/CPD-P2P.txt', 'a')
		f.write(result_string)
		f.close()

		fig.suptitle('Patient ' + str(i + 1) + ' - MR to US')
		plt.show()
		plt.savefig('./prostate_results-' + fname + '/CPD-P2P-' + str(i + 1) + '.png', dpi=300)
		plt.close()
	'''

def get_filenames(data):
	filenames = []
	for fname in data['Filenames'][0]: # Get all filenames.
		path = str(fname[0][94:-4])
		path = path.split('-')
		filenames.append([path[0], path[1]])
	return filenames

def get_indices(data, filenames):
	indxs = []
	for i in range(len(filenames) - 2): # Get all complete triples of scans.
		pid = filenames[i][0]
		if pid == filenames[i + 1][0] and pid == filenames[i + 2][0]:
			if re.match('.*ADC$', filenames[i][1]) and re.match('^t1', filenames[i + 1][1]) and re.match('^t2', filenames[i + 2][1]):
				if data['Meshes'][0][i].shape[1] >= 4:
					indxs.append(i)
					indxs.append(i + 1)
					indxs.append(i + 2)
	return indxs

def get_prostate_data(data, indxs, dims=[2048,3]):
	all_prostates = []
	for indx in indxs:
		prostate = data['Meshes'][0][indx] # Gets to the list of structs.
		prostate_j = [np.array([])] * 8
		ref_data = []

		for i in range(prostate.shape[1]): # Gets to the individual dataset and gets the reference normalization parameters.

			if prostate[0][i][0][0][0].size > 0:
				if prostate[0][i][0][0][0] == 'ROI 1':
				
					all_shapes = [prostate[0][i][0][0][x].shape for x in range(len(prostate[0][i][0][0]))]
					threeD_shapes = [all_shapes[x] for x in range(len(all_shapes)) if 3 in all_shapes[x]]
					second_largest = [threeD_shapes.index(x) for x in sorted(threeD_shapes, key=lambda y: y[1],  reverse=True)][1]
					val_index = all_shapes.index(threeD_shapes[second_largest])
					prostate_j[7] = prostate[0][i][0][0][val_index]
					
					if prostate_j[7].shape[0] > dims[0]:
						prostate_j[7] = prostate_j[7][np.random.randint(prostate_j[7].shape[0], size=dims[0]), :]
					mc_data = prostate_j[7]
					norm_data = mc_data - np.mean(mc_data, axis=0)

		for j in range(prostate.shape[1]): # Gets to the individual dataset and stores rescaled, upsampled and centered points.

			if prostate[0][j][0][0][0].size > 0:

				all_shapes = [prostate[0][j][0][0][x].shape for x in range(len(prostate[0][j][0][0]))]
				threeD_shapes = [all_shapes[x] for x in range(len(all_shapes)) if 3 in all_shapes[x]]
				try:
					val_index = all_shapes.index((1,3))
				except ValueError:
					second_largest = [threeD_shapes.index(x) for x in sorted(threeD_shapes, key=lambda y: y[1],  reverse=True)][1]
					val_index = all_shapes.index(threeD_shapes[second_largest])
				prostate_j_data = prostate[0][j][0][0][val_index]

				if prostate_j_data.shape[0] > dims[0]:				prostate_j_data_centered = prostate_j_data - np.mean(mc_data, axis=0)
				prostate_j_data_normalized = 2 * (prostate_j_data_centered - np.min(norm_data)) / np.ptp(norm_data) - 1

				prostate_j_data = prostate_j_data[np.random.randint(prostate_j_data.shape[0], size=dims[0]), :]
				prostate_j_data_resized = np.resize(prostate_j_data_normalized, dims)

				if prostate[0][j][0][0][0] == 'ROI 1':
					prostate_j[0] = prostate_j_data_resized
				elif prostate[0][j][0][0][0] == 'ROI 2':
					prostate_j[1] = prostate_j_data_resized
				elif prostate[0][j][0][0][0] == 'ROI 3':
					prostate_j[2] = prostate_j_data_resized
				elif prostate[0][j][0][0][0] == 'ROI 4':
					prostate_j[3] = prostate_j_data_resized
				elif prostate[0][j][0][0][0] == 'ROI 5':
					prostate_j[4] = prostate_j_data_resized
				elif prostate[0][j][0][0][0] == 'Apex':
					prostate_j[5] = prostate_j_data_resized
				elif prostate[0][j][0][0][0] == 'Base':
					prostate_j[6] = prostate_j_data_resized

		all_prostates.append(prostate_j)
	return all_prostates

def predict_prostate_file(fname):

	if not os.path.exists('./prostate_results-' + fname + '/'):
		os.mkdir('./prostate_results-' + fname + '/')

	header_string = 'P_ID\tMOVING\tFIXED\tTIME\tDC_P\tDH_P\tDC_Tz\tDH_Tz\tDC_T1\tDH_T1\tDC_T2\tDH_T2\tDC_T3\tDH_T3\tD_Apex\tD_Base\n'
	f = open('./prostate_results-' + fname + '/P2P.txt', 'a')
	f.write(header_string)
	f.close()
	f = open('./prostate_results-' + fname + '/PTz2PTz.txt', 'a')
	f.write(header_string)
	f.close()
	'''
	f = open('./prostate_results-' + fname + '/CPD-P2P.txt', 'a')
	f.write(header_string)
	f.close()
	f = open('./prostate_results-' + fname + '/CPD-PTz2PTz.txt', 'a')
	f.write(header_string)
	f.close()
	'''

	# Load the model.
	model = load_model(fname + '.h5',
				   	   custom_objects={'MatMul':MatMul},
				       compile=False)
	
	dims = [2048, 3]
	prostate_data = sio.loadmat('prostate.mat')
	filenames = get_filenames(prostate_data)
	indxs  = get_indices(prostate_data, filenames)
	all_prostates = get_prostate_data(prostate_data, indxs)
	max_iters = len(all_prostates) - 2

	# All possible patient data permuatations for the testing:
	# ADC <- T1, ADC <- T2, T1 <- ADC, T1 <- T2, T2 <- ADC, T2 <- T1
	perms = [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]]
	tags = ['ADC', 'T1', 'T2']

	# CTN - Contour to Contour
	for i in range(0, max_iters, 3):

		ADC = all_prostates[i]
		T1 = all_prostates[i+1]
		T2 = all_prostates[i+2]
		patient_data = [ADC, T1, T2]

		for perm in perms:

			# Fixed Prostate
			fixed_prostate = patient_data[perm[0]][0]
			fixed_prostate_u = np.unique(fixed_prostate, axis=0)
			x_fp, y_fp, z_fp = get_unique_plot_points(fixed_prostate)
			# Fixed Tz
			fixed_transition_zone = patient_data[perm[0]][1]
			fixed_transition_zone_u = np.unique(fixed_transition_zone, axis=0)
			x_ftz, y_ftz, z_ftz = get_unique_plot_points(fixed_transition_zone)
			# Fixed Apex & Base
			fixed_apex = patient_data[perm[0]][5]
			fixed_apex_u = np.unique(fixed_apex, axis=0)
			x_fa, y_fa, z_fa = get_unique_plot_points(fixed_apex)
			fixed_base = patient_data[perm[0]][6]
			fixed_base_u = np.unique(fixed_base, axis=0)
			x_fb, y_fb, z_fb = get_unique_plot_points(fixed_base)
			# Fixed Tumor(s)
			fixed_t1 = patient_data[perm[0]][2]
			if fixed_t1.size > 0:
				fixed_t1_u = np.unique(fixed_t1, axis=0)
				x_ft1, y_ft1, z_ft1 = get_unique_plot_points(fixed_t1)
			fixed_t2 = patient_data[perm[0]][3]
			if fixed_t2.size > 0:
				fixed_t2_u = np.unique(fixed_t2, axis=0)
				x_ft2, y_ft2, z_ft2 = get_unique_plot_points(fixed_t2)
			fixed_t3 = patient_data[perm[0]][4]
			if fixed_t3.size > 0:
				fixed_t3_u = np.unique(fixed_t3, axis=0)
				x_ft3, y_ft3, z_ft3 = get_unique_plot_points(fixed_t3)

			# Moving Prostate
			moving_prostate = patient_data[perm[1]][0]
			x_mp, y_mp, z_mp = get_unique_plot_points(moving_prostate)
			# Moving Tz
			moving_transition_zone = patient_data[perm[1]][1]
			x_mtz, y_mtz, z_mtz = get_unique_plot_points(moving_transition_zone)
			# Moving Apex & Base
			moving_apex = patient_data[perm[1]][5]
			x_ma, y_ma, z_ma = get_unique_plot_points(moving_apex)
			moving_base = patient_data[perm[1]][6]
			x_mb, y_mb, z_mb = get_unique_plot_points(moving_base)
			# Moving Tumor(s)
			moving_t1 = patient_data[perm[1]][2]
			if moving_t1.size > 0:
				moving_t1_u = np.unique(moving_t1, axis=0)
				x_mt1, y_mt1, z_mt1 = get_unique_plot_points(moving_t1)
			moving_t2 = patient_data[perm[1]][3]
			if moving_t2.size > 0:
				moving_t2_u = np.unique(moving_t2, axis=0)
				x_mt2, y_mt2, z_mt2 = get_unique_plot_points(moving_t2)
			moving_t3 = patient_data[perm[1]][4]
			if moving_t3.size > 0:
				moving_t3_u = np.unique(moving_t3, axis=0)
				x_mt3, y_mt3, z_mt3 = get_unique_plot_points(moving_t3)

			# Moving2Fixed Prostate
			t = time.time()
			pred = moving_prostate
			#pred = model.predict([[np.array(fixed_prostate)],
			#					  [np.array(moving_prostate)],
			#					  [np.array(moving_prostate)]])
			t = round(time.time() - t, 3)
			pred_u = np.unique(pred, axis=0)
			x_pred, y_pred, z_pred = get_unique_plot_points(pred)
			#pred_u = np.unique(pred[0], axis=0)
			#x_pred, y_pred, z_pred = get_unique_plot_points(pred[0])
			# Moving2Fixed Tz
			pred_transition_zone = moving_transition_zone
			#pred_transition_zone = model.predict([[np.array(fixed_prostate)],
			#						   			  [np.array(moving_prostate)],
			#						   			  [np.array(moving_transition_zone)]])
			pred_transition_zone_u = np.unique(pred_transition_zone, axis=0)
			x_ptz, y_ptz, z_ptz = get_unique_plot_points(pred_transition_zone)
			#pred_transition_zone_u = np.unique(pred_transition_zone[0], axis=0)
			#x_ptz, y_ptz, z_ptz = get_unique_plot_points(pred_transition_zone[0])
			# Moving2Fixed Apex & Base
			pred_apex = moving_apex
			#pred_apex = model.predict([[np.array(fixed_prostate)],
			#						   [np.array(moving_prostate)],
			#						   [np.array(moving_apex)]])
			pred_apex_u = np.unique(pred_apex, axis=0)
			x_pa, y_pa, z_pa = get_unique_plot_points(pred_apex)
			#pred_apex_u = np.unique(pred_apex[0], axis=0)
			#x_pa, y_pa, z_pa = get_unique_plot_points(pred_apex[0])
			pred_base = moving_base
			#pred_base = model.predict([[np.array(fixed_prostate)],
			#						   [np.array(moving_prostate)],
			#						   [np.array(moving_base)]])
			pred_base_u = np.unique(pred_base, axis=0)
			x_pb, y_pb, z_pb = get_unique_plot_points(pred_base)
			#pred_base_u = np.unique(pred_base[0], axis=0)
			#x_pb, y_pb, z_pb = get_unique_plot_points(pred_base[0])
			# Moving2Fixed Tumor(s)
			if moving_t1.size > 0 and fixed_t1.size > 0:
				pred_t1 = moving_t1
				#pred_t1 = model.predict([[np.array(fixed_prostate)],
				#						  [np.array(moving_prostate)],
				#						  [np.array(moving_t1)]])
				pred_t1_u = np.unique(pred_t1, axis=0)
				x_pt1, y_pt1, z_pt1 = get_unique_plot_points(pred_t1)
				#pred_t1_u = np.unique(pred_t1[0], axis=0)
				#x_pt1, y_pt1, z_pt1 = get_unique_plot_points(pred_t1[0])
			if moving_t2.size > 0 and fixed_t2.size > 0:
				pred_t2 = moving_t2
				#pred_t2 = model.predict([[np.array(fixed_prostate)],
				#						  [np.array(moving_prostate)],
				#						  [np.array(moving_t2)]])
				pred_t2_u = np.unique(pred_t2, axis=0)
				x_pt2, y_pt2, z_pt2 = get_unique_plot_points(pred_t2)
				#pred_t2_u = np.unique(pred_t2[0], axis=0)
				#x_pt2, y_pt2, z_pt2 = get_unique_plot_points(pred_t2[0])
			if moving_t3.size > 0 and fixed_t3.size > 0:
				pred_t3 = moving_t3
				#pred_t3 = model.predict([[np.array(fixed_prostate)],
				#						  [np.array(moving_prostate)],
				#						  [np.array(moving_t3)]])
				pred_t3_u = np.unique(pred_t3, axis=0)
				x_pt3, y_pt3, z_pt3 = get_unique_plot_points(pred_t3)
				#pred_t3_u = np.unique(pred_t3[0], axis=0)
				#x_pt3, y_pt3, z_pt3 = get_unique_plot_points(pred_t3[0])

			fig = plt.figure()
			ax0 = fig.add_subplot(221, projection='3d')
			ax0.set_title('Fixed P & Fixed Tz')
			ax0.scatter(x_fp, y_fp, z_fp, c='y', marker='.', alpha=0.2)
			ax0.scatter(x_ftz, y_ftz, z_ftz, c='m', marker='.', alpha=0.2)
			ax0.scatter(x_fa, y_fa, z_fa, c='k', marker='^', alpha=1)
			ax0.scatter(x_fb, y_fb, z_fb, c='k', marker='v', alpha=1)
			if moving_t1.size > 0 and fixed_t1.size > 0:
				ax0.scatter(x_ft1, y_ft1, z_ft1, c='k', marker='.', alpha=0.2)		
			if moving_t2.size > 0 and fixed_t2.size > 0:
				ax0.scatter(x_ft2, y_ft2, z_ft2, c='k', marker='.', alpha=0.2)					
			if moving_t3.size > 0 and fixed_t3.size > 0:
				ax0.scatter(x_ft3, y_ft3, z_ft3, c='k', marker='.', alpha=0.2)		

			ax1 = fig.add_subplot(222, projection='3d')
			ax1.set_title('Moving P & Moving Tz')
			ax1.scatter(x_mp, y_mp, z_mp, c='r', marker='.', alpha=0.2)
			ax1.scatter(x_mtz, y_mtz, z_mtz, c='b', marker='.', alpha=0.2)
			ax1.scatter(x_ma, y_ma, z_ma, c='k', marker='^', alpha=1)
			ax1.scatter(x_mb, y_mb, z_mb, c='k', marker='v', alpha=1)
			if moving_t1.size > 0 and fixed_t1.size > 0:
				ax1.scatter(x_mt1, y_mt1, z_mt1, c='k', marker='.', alpha=0.2)		
			if moving_t2.size > 0 and fixed_t2.size > 0:
				ax1.scatter(x_mt2, y_mt2, z_mt2, c='k', marker='.', alpha=0.2)					
			if moving_t3.size > 0 and fixed_t3.size > 0:
				ax1.scatter(x_mt3, y_mt3, z_mt3, c='k', marker='.', alpha=0.2)	

			ax2 = fig.add_subplot(223, projection='3d')
			ax2.set_title('Fixed P & Moving P')
			#ax2.set_title('Fixed P & Moving2Fixed P')
			ax2.scatter(x_fp, y_fp, z_fp, c='y', marker='.', alpha=0.2)
			ax2.scatter(x_pred, y_pred, z_pred, c='g', marker='.', alpha=0.2)
			ax2.scatter(x_fa, y_fa, z_fa, c='k', marker='^', alpha=1)
			ax2.scatter(x_fb, y_fb, z_fb, c='k', marker='v', alpha=1)
			ax2.scatter(x_pa, y_pa, z_pa, c='b', marker='^', alpha=1)
			ax2.scatter(x_pb, y_pb, z_pb, c='b', marker='v', alpha=1)
			if moving_t1.size > 0 and fixed_t1.size > 0:
				ax2.scatter(x_ft1, y_ft1, z_ft1, c='k', marker='.', alpha=0.2)		
				ax2.scatter(x_pt1, y_pt1, z_pt1, c='b', marker='.', alpha=0.2)		
			if moving_t2.size > 0 and fixed_t2.size > 0:
				ax2.scatter(x_ft2, y_ft2, z_ft2, c='k', marker='.', alpha=0.2)		
				ax2.scatter(x_pt2, y_pt2, z_pt2, c='b', marker='.', alpha=0.2)					
			if moving_t3.size > 0 and fixed_t3.size > 0:
				ax2.scatter(x_ft2, y_ft2, z_ft2, c='k', marker='.', alpha=0.2)		
				ax2.scatter(x_pt3, y_pt3, z_pt3, c='b', marker='.', alpha=0.2)	

			ax3 = fig.add_subplot(224, projection='3d')
			ax3.set_title('Fixed Tz & Moving Tz')
			#ax3.set_title('Fixed Tz & Moving2Fixed Tz')
			ax3.scatter(x_ftz, y_ftz, z_ftz, c='m', marker='.', alpha=0.2)
			ax3.scatter(x_ptz, y_ptz, z_ptz, c='g', marker='.', alpha=0.2)

			set_plot_ax_lims([ax0, ax1, ax2, ax3])

			ref_data = patient_data[perm[0]][7]
			ref_data_mc = ref_data - np.mean(ref_data, axis=0)

			# Scale the data so we can compute metrics with correct values.
			pred_dn = denormalize(pred_u, ref_data_mc)
			fixed_prostate_dn = denormalize(fixed_prostate_u, ref_data_mc)
			d_c_p = round(chamfer(fixed_prostate_dn, pred_dn), 3)
			d_h_p = round(hausdorff_2way(fixed_prostate_dn, pred_dn), 3)

			pred_transition_zone_dn = denormalize(pred_transition_zone_u, ref_data_mc)
			fixed_transition_zone_dn = denormalize(fixed_transition_zone_u, ref_data_mc)
			d_c_tz = round(chamfer(fixed_transition_zone_dn, pred_transition_zone_dn), 3)
			d_h_tz = round(hausdorff_2way(fixed_transition_zone_dn, pred_transition_zone_dn), 3)

			fixed_apex_dn = denormalize(fixed_apex_u, ref_data_mc)
			fixed_base_dn = denormalize(fixed_base_u, ref_data_mc)
			pred_apex_dn = denormalize(pred_apex_u, ref_data_mc)
			pred_base_dn = denormalize(pred_base_u, ref_data_mc)
			d_apex = round(np.linalg.norm(fixed_apex_dn - pred_apex_dn), 3)
			d_base = round(np.linalg.norm(fixed_base_dn - pred_base_dn), 3)

			if moving_t1.size > 0 and fixed_t1.size > 0:
				ft1_dn = denormalize(fixed_t1_u, ref_data_mc)
				pt1_dn = denormalize(pred_t1_u, ref_data_mc)
				d_c_t1 = round(chamfer(ft1_dn, pt1_dn), 3)
				d_h_t1 = round(hausdorff_2way(ft1_dn, pt1_dn), 3)
			else:
				d_c_t1 = ' '
				d_h_t1 = ' '
			if moving_t2.size > 0 and fixed_t2.size > 0:
				ft2_dn = denormalize(fixed_t2_u, ref_data_mc)
				pt2_dn = denormalize(pred_t2_u, ref_data_mc)
				d_c_t2 = round(chamfer(ft2_dn, pt2_dn), 3)
				d_h_t2 = round(hausdorff_2way(ft2_dn, pt2_dn), 3)
			else:
				d_c_t2 = ' '
				d_h_t2 = ' '
			if moving_t3.size > 0 and fixed_t3.size > 0:
				ft3_dn = denormalize(fixed_t3_u, ref_data_mc)
				pt3_dn = denormalize(pred_t3_u, ref_data_mc)
				d_c_t3 = round(chamfer(ft3_dn, pt3_dn), 3)
				d_h_t3 = round(hausdorff_2way(ft3_dn, pt3_dn), 3)
			else:
				d_c_t3 = ' '
				d_h_t3 = ' '

			fig.suptitle(str(filenames[indxs[i]][0]) + ' ' + str(tags[perm[1]]) + ' to ' + str(tags[perm[0]]))
			plt.show()
			plt.savefig('./prostate_results-' + fname + '/P2P-' + str(filenames[indxs[i]][0]) + '_' + str(tags[perm[1]]) + '-to-' + str(tags[perm[0]]) + '.png', dpi=300)
			plt.close()

			result_string = str(filenames[indxs[i]][0]) + '\t' + str(tags[perm[1]]) + '\t' + str(tags[perm[0]]) + '\t' + str(t) + '\t' + \
								str(d_c_p) + '\t' + str(d_h_p) + '\t' + \
								str(d_c_tz) + '\t' + str(d_h_tz) + '\t' + \
								str(d_c_t1) + '\t' + str(d_h_t1) + '\t' + \
								str(d_c_t2) + '\t' + str(d_h_t2) + '\t' + \
								str(d_c_t3) + '\t' + str(d_h_t3) + '\t' + \
								str(d_apex) + '\t' + str(d_base) + '\n'
			f = open('./prostate_results-' + fname + '/P2P.txt', 'a')
			f.write(result_string)
			f.close()

	'''
	# CTN - Contour&Tz to Contour&Tz
	for i in range(0, max_iters, 3):

		ADC = all_prostates[i]
		T1 = all_prostates[i+1]
		T2 = all_prostates[i+2]
		patient_data = [ADC, T1, T2]
		
		for perm in perms:

			# Fixed Prostate & Tz
			fixed = np.resize(np.vstack((np.unique(patient_data[perm[0]][0], axis=0), np.unique(patient_data[perm[0]][1], axis=0))), dims)
			fixed_u = np.unique(fixed, axis=0)
			x_fptz, y_fptz, z_fptz = get_unique_plot_points(fixed)
			# Fixed Tz
			fixed_transition_zone = patient_data[perm[0]][1]
			fixed_transition_zone_u = np.unique(fixed_transition_zone, axis=0)
			x_ftz, y_ftz, z_ftz = get_unique_plot_points(fixed_transition_zone)
			# Fixed Apex & Base
			fixed_apex = patient_data[perm[0]][5]
			fixed_apex_u = np.unique(fixed_apex, axis=0)
			x_fa, y_fa, z_fa = get_unique_plot_points(fixed_apex)
			fixed_base = patient_data[perm[0]][6]
			fixed_base_u = np.unique(fixed_base, axis=0)
			x_fb, y_fb, z_fb = get_unique_plot_points(fixed_base)

			# Moving Prostate & Tz
			moving = np.resize(np.vstack((np.unique(patient_data[perm[1]][0], axis=0), np.unique(patient_data[perm[1]][1], axis=0))), dims)
			moving_u = np.unique(moving, axis=0)
			x_mptz, y_mptz, z_mptz = get_unique_plot_points(moving)
			# Moving Tz
			moving_transition_zone = patient_data[perm[1]][1]
			x_mtz, y_mtz, z_mtz = get_unique_plot_points(moving_transition_zone)
			# Moving Apex & Base
			moving_apex = patient_data[perm[1]][5]
			x_ma, y_ma, z_ma = get_unique_plot_points(moving_apex)
			moving_base = patient_data[perm[1]][6]
			x_mb, y_mb, z_mb = get_unique_plot_points(moving_base)

			# Moving2Fixed Prostate & Tz
			t = time.time()
			pred = model.predict([[np.array(fixed)],
								  [np.array(moving)],
								  [np.array(moving)]])
			t = round(time.time() - t, 3)
			pred_u = np.unique(pred[0], axis=0)
			x_pred, y_pred, z_pred = get_unique_plot_points(pred[0])
			# Moving2Fixed Tz
			pred_transition_zone = model.predict([[np.array(fixed)],
									   			  [np.array(moving)],
									   			  [np.array(moving_transition_zone)]])
			pred_transition_zone_u = np.unique(pred_transition_zone[0], axis=0)
			x_ptz, y_ptz, z_ptz = get_unique_plot_points(pred_transition_zone[0])
			# Moving2Fixed Apex & Base
			pred_apex = model.predict([[np.array(fixed)],
									   [np.array(moving)],
									   [np.array(moving_apex)]])
			pred_apex_u = np.unique(pred_apex[0], axis=0)
			x_pa, y_pa, z_pa = get_unique_plot_points(pred_apex[0])
			pred_base = model.predict([[np.array(fixed)],
									   [np.array(moving)],
									   [np.array(moving_base)]])
			pred_base_u = np.unique(pred_base[0], axis=0)
			x_pb, y_pb, z_pb = get_unique_plot_points(pred_base[0])

			fig = plt.figure()
			ax0 = fig.add_subplot(221, projection='3d')
			ax0.set_title('Fixed P & Tz')
			ax0.scatter(x_fptz, y_fptz, z_fptz, c='y', marker='.', alpha=0.2)
			ax0.scatter(x_fa, y_fa, z_fa, c='k', marker='^', alpha=1)
			ax0.scatter(x_fb, y_fb, z_fb, c='k', marker='v', alpha=1)

			ax1 = fig.add_subplot(222, projection='3d')
			ax1.set_title('Moving P & Tz')
			ax1.scatter(x_mptz, y_mptz, z_mptz, c='r', marker='.', alpha=0.2)
			ax1.scatter(x_ma, y_ma, z_ma, c='k', marker='^', alpha=1)
			ax1.scatter(x_mb, y_mb, z_mb, c='k', marker='v', alpha=1)

			ax2 = fig.add_subplot(223, projection='3d')
			ax2.set_title('Moving2Fixed P & Tz')
			ax2.scatter(x_fptz, y_fptz, z_fptz, c='y', marker='.', alpha=0.2)
			ax2.scatter(x_pred, y_pred, z_pred, c='g', marker='.', alpha=0.2)
			ax2.scatter(x_fa, y_fa, z_fa, c='k', marker='^', alpha=1)
			ax2.scatter(x_fb, y_fb, z_fb, c='k', marker='v', alpha=1)
			ax2.scatter(x_pa, y_pa, z_pa, c='b', marker='^', alpha=1)
			ax2.scatter(x_pb, y_pb, z_pb, c='b', marker='v', alpha=1)

			ax3 = fig.add_subplot(224, projection='3d')
			ax3.set_title('Fixed Tz & Moving2Fixed Tz')
			ax3.scatter(x_ftz, y_ftz, z_ftz, c='m', marker='.', alpha=0.2)
			ax3.scatter(x_ptz, y_ptz, z_ptz, c='g', marker='.', alpha=0.2)

			set_plot_ax_lims([ax0, ax1, ax2, ax3])

			ref_data = patient_data[perm[0]][7]
			ref_data_mc = ref_data - np.mean(ref_data, axis=0)

			# Scale the data so we can compute metrics with correct values.
			pred_dn = denormalize(pred_u, ref_data_mc)
			fixed_dn = denormalize(fixed_u, ref_data_mc)
			d_c_p = round(chamfer(fixed_dn, pred_dn), 3)
			d_h_p = round(hausdorff_2way(fixed_dn, pred_dn), 3)

			pred_transition_zone_dn = denormalize(pred_transition_zone_u, ref_data_mc)
			fixed_transition_zone_dn = denormalize(fixed_transition_zone_u, ref_data_mc)
			d_c_tz = round(chamfer(fixed_transition_zone_dn, pred_transition_zone_dn), 3)
			d_h_tz = round(hausdorff_2way(fixed_transition_zone_dn, pred_transition_zone_dn), 3)

			fixed_apex_dn = denormalize(fixed_apex_u, ref_data_mc)
			fixed_base_dn = denormalize(fixed_base_u, ref_data_mc)
			pred_apex_dn = denormalize(pred_apex_u, ref_data_mc)
			pred_base_dn = denormalize(pred_base_u, ref_data_mc)
			d_apex = round(np.linalg.norm(fixed_apex_dn - pred_apex_dn), 3)
			d_base = round(np.linalg.norm(fixed_base_dn - pred_base_dn), 3)

			fig.suptitle(str(filenames[indxs[i]][0]) + ' ' + str(tags[perm[1]]) + ' to ' + str(tags[perm[0]]))
			plt.savefig('./prostate_results-' + fname + '/PTz2PTz-' + str(filenames[indxs[i]][0]) + '_' + str(tags[perm[1]]) + '-to-' + str(tags[perm[0]]) + '.png', dpi=300)
			plt.close()

			result_string = str(filenames[indxs[i]][0]) + '\t' + str(tags[perm[1]]) + '\t' + str(tags[perm[0]]) + '\t' + str(t) + '\t' + str(d_c_p) + '\t' + str(d_h_p) + '\t' + str(d_c_tz) + '\t' + str(d_h_tz) + '\t' + str(d_apex) + '\t' + str(d_base) + '\n'
			f = open('./prostate_results-' + fname + '/PTz2PTz.txt', 'a')
			f.write(result_string)
			f.close()

	# CPD - Contour to Contour
	for i in range(0, max_iters, 3):

		ADC = all_prostates[i]
		T1 = all_prostates[i+1]
		T2 = all_prostates[i+2]
		patient_data = [ADC, T1, T2]
		
		for perm in perms:

			# Fixed Prostate
			fixed_prostate = patient_data[perm[0]][0]
			fixed_prostate_u = np.unique(fixed_prostate, axis=0)
			x_fp, y_fp, z_fp = get_unique_plot_points(fixed_prostate)
			# Fixed Tz
			fixed_transition_zone = patient_data[perm[0]][1]
			fixed_transition_zone_u = np.unique(fixed_transition_zone, axis=0)
			x_ftz, y_ftz, z_ftz = get_unique_plot_points(fixed_transition_zone)
			# Fixed Apex & Base
			fixed_apex = patient_data[perm[0]][5]
			fixed_apex_u = np.unique(fixed_apex, axis=0)
			x_fa, y_fa, z_fa = get_unique_plot_points(fixed_apex)
			fixed_base = patient_data[perm[0]][6]
			fixed_base_u = np.unique(fixed_base, axis=0)
			x_fb, y_fb, z_fb = get_unique_plot_points(fixed_base)

			# Moving Prostate
			moving_prostate = patient_data[perm[1]][0]
			moving_prostate_u = np.unique(moving_prostate, axis=0)
			x_mp, y_mp, z_mp = get_unique_plot_points(moving_prostate)
			# Moving Tz
			moving_transition_zone = patient_data[perm[1]][1]
			x_mtz, y_mtz, z_mtz = get_unique_plot_points(moving_transition_zone)
			# Moving Apex & Base
			moving_apex = patient_data[perm[1]][5]
			x_ma, y_ma, z_ma = get_unique_plot_points(moving_apex)
			moving_base = patient_data[perm[1]][6]
			x_mb, y_mb, z_mb = get_unique_plot_points(moving_base)

			# Moving2Fixed Prostate
			reg = deformable_registration(**{'X':fixed_prostate, 'Y':moving_prostate, 'max_iterations':50})
			t = time.time()
			pred, params = reg.register()
			t = round(time.time() - t, 3)
			pred_u = np.unique(pred, axis=0)
			x_pred, y_pred, z_pred = get_unique_plot_points(pred)
			# Moving2Fixed Tz
			moving_transition_zone_u = np.unique(moving_transition_zone, axis=0)
			pred_transition_zone = moving_transition_zone_u + np.dot(gaussian_kernel(moving_prostate, moving_transition_zone_u), params[1])
			pred_transition_zone_u = np.unique(pred_transition_zone, axis=0)
			x_ptz, y_ptz, z_ptz = get_unique_plot_points(pred_transition_zone)
			# Moving2Fixed Apex & Base
			moving_apex_u = np.unique(moving_apex, axis=0)
			pred_apex = moving_apex_u + np.dot(gaussian_kernel(moving_prostate, moving_apex_u), params[1])
			x_pa, y_pa, z_pa = get_unique_plot_points(pred_apex)
			moving_base_u = np.unique(moving_base, axis=0)
			pred_base = moving_base_u + np.dot(gaussian_kernel(moving_prostate, moving_base_u), params[1])
			x_pb, y_pb, z_pb = get_unique_plot_points(pred_base)

			fig = plt.figure()
			ax0 = fig.add_subplot(221, projection='3d')
			ax0.set_title('Fixed P & Fixed Tz')
			ax0.scatter(x_fp, y_fp, z_fp, c='y', marker='.', alpha=0.2)
			ax0.scatter(x_ftz, y_ftz, z_ftz, c='m', marker='.', alpha=0.2)
			ax0.scatter(x_fa, y_fa, z_fa, c='k', marker='^', alpha=1)
			ax0.scatter(x_fb, y_fb, z_fb, c='k', marker='v', alpha=1)

			ax1 = fig.add_subplot(222, projection='3d')
			ax1.set_title('Moving P & Moving Tz')
			ax1.scatter(x_mp, y_mp, z_mp, c='r', marker='.', alpha=0.2)
			ax1.scatter(x_mtz, y_mtz, z_mtz, c='b', marker='.', alpha=0.2)
			ax1.scatter(x_ma, y_ma, z_ma, c='k', marker='^', alpha=1)
			ax1.scatter(x_mb, y_mb, z_mb, c='k', marker='v', alpha=1)

			ax2 = fig.add_subplot(223, projection='3d')
			ax2.set_title('Fixed P & Moving2Fixed P')
			ax2.scatter(x_fp, y_fp, z_fp, c='y', marker='.', alpha=0.2)
			ax2.scatter(x_pred, y_pred, z_pred, c='g', marker='.', alpha=0.2)
			ax2.scatter(x_fa, y_fa, z_fa, c='k', marker='^', alpha=1)
			ax2.scatter(x_fb, y_fb, z_fb, c='k', marker='v', alpha=1)
			ax2.scatter(x_pa, y_pa, z_pa, c='b', marker='^', alpha=1)
			ax2.scatter(x_pb, y_pb, z_pb, c='b', marker='v', alpha=1)

			ax3 = fig.add_subplot(224, projection='3d')
			ax3.set_title('Fixed Tz & Moving2Fixed Tz')
			ax3.scatter(x_ftz, y_ftz, z_ftz, c='m', marker='.', alpha=0.2)
			ax3.scatter(x_ptz, y_ptz, z_ptz, c='g', marker='.', alpha=0.2)

			set_plot_ax_lims([ax0, ax1, ax2, ax3])

			ref_data = patient_data[perm[0]][7]
			ref_data_mc = ref_data - np.mean(ref_data, axis=0)

			# Scale the data so we can compute metrics with correct values.
			pred_dn = denormalize(pred_u, ref_data_mc)
			fixed_prostate_dn = denormalize(fixed_prostate_u, ref_data_mc)
			d_c_p = round(chamfer(fixed_prostate_dn, pred_dn), 3)
			d_h_p = round(hausdorff_2way(fixed_prostate_dn, pred_dn), 3)

			pred_transition_zone_dn = denormalize(pred_transition_zone_u, ref_data_mc)
			fixed_transition_zone_dn = denormalize(fixed_transition_zone_u, ref_data_mc)
			d_c_tz = round(chamfer(fixed_transition_zone_dn, pred_transition_zone_dn), 3)
			d_h_tz = round(hausdorff_2way(fixed_transition_zone_dn, pred_transition_zone_dn), 3)

			fixed_apex_dn = denormalize(fixed_apex_u, ref_data_mc)
			fixed_base_dn = denormalize(fixed_base_u, ref_data_mc)
			pred_apex_dn = denormalize(pred_apex, ref_data_mc)
			pred_base_dn = denormalize(pred_base, ref_data_mc)
			d_apex = round(np.linalg.norm(fixed_apex_dn - pred_apex_dn), 3)
			d_base = round(np.linalg.norm(fixed_base_dn - pred_base_dn), 3)

			fig.suptitle(str(filenames[indxs[i]][0]) + ' ' + str(tags[perm[1]]) + ' to ' + str(tags[perm[0]]))
			plt.show()
			plt.savefig('./prostate_results/CPD-P2P-' + str(filenames[indxs[i]][0]) + '_' + str(tags[perm[1]]) + '-to-' + str(tags[perm[0]]) + '.png', dpi=300)
			plt.close()

			result_string = str(filenames[indxs[i]][0]) + '\t' + str(tags[perm[1]]) + '\t' + str(tags[perm[0]]) + '\t' + str(t) + '\t' + str(d_c_p) + '\t' + str(d_h_p) + '\t' + str(d_c_tz) + '\t' + str(d_h_tz) + '\t' + str(d_apex) + '\t' + str(d_base) + '\n'
			f = open('./prostate_results/CPD-P2P.txt', 'a')
			f.write(result_string)
			f.close()

	# CPD - Contour&Tz to Contour&Tz
	for i in range(0, max_iters, 3):

		ADC = all_prostates[i]
		T1 = all_prostates[i+1]
		T2 = all_prostates[i+2]
		patient_data = [ADC, T1, T2]
		
		for perm in perms:

			# Fixed Prostate & Tz
			fixed = np.resize(np.vstack((np.unique(patient_data[perm[0]][0], axis=0), np.unique(patient_data[perm[0]][1], axis=0))), dims)
			fixed_u = np.unique(fixed, axis=0)
			x_fptz, y_fptz, z_fptz = get_unique_plot_points(fixed)
			# Fixed Tz
			fixed_transition_zone = patient_data[perm[0]][1]
			fixed_transition_zone_u = np.unique(fixed_transition_zone, axis=0)
			x_ftz, y_ftz, z_ftz = get_unique_plot_points(fixed_transition_zone)
			# Fixed Apex & Base
			fixed_apex = patient_data[perm[0]][5]
			fixed_apex_u = np.unique(fixed_apex, axis=0)
			x_fa, y_fa, z_fa = get_unique_plot_points(fixed_apex)
			fixed_base = patient_data[perm[0]][6]
			fixed_base_u = np.unique(fixed_base, axis=0)
			x_fb, y_fb, z_fb = get_unique_plot_points(fixed_base)

			# Moving Prostate & Tz
			moving = np.resize(np.vstack((np.unique(patient_data[perm[1]][0], axis=0), np.unique(patient_data[perm[1]][1], axis=0))), dims)
			moving_u = np.unique(moving, axis=0)
			x_mptz, y_mptz, z_mptz = get_unique_plot_points(moving)
			# Moving Tz
			moving_transition_zone = patient_data[perm[1]][1]
			x_mtz, y_mtz, z_mtz = get_unique_plot_points(moving_transition_zone)
			# Moving Apex & Base
			moving_apex = patient_data[perm[1]][5]
			x_ma, y_ma, z_ma = get_unique_plot_points(moving_apex)
			moving_base = patient_data[perm[1]][6]
			x_mb, y_mb, z_mb = get_unique_plot_points(moving_base)

			# Moving2Fixed Prostate & Tz
			reg = deformable_registration(**{'X':fixed, 'Y':moving, 'max_iterations':50})
			t = time.time()
			pred, params = reg.register()
			t = round(time.time() - t, 3)
			pred_u = np.unique(pred, axis=0)
			x_pred, y_pred, z_pred = get_unique_plot_points(pred)
			# Moving2Fixed Tz
			moving_transition_zone_u = np.unique(moving_transition_zone, axis=0)
			pred_transition_zone = moving_transition_zone_u + np.dot(gaussian_kernel(moving, moving_transition_zone_u), params[1])
			pred_transition_zone_u = np.unique(pred_transition_zone, axis=0)
			x_ptz, y_ptz, z_ptz = get_unique_plot_points(pred_transition_zone)
			# Moving2Fixed Apex & Base
			moving_apex_u = np.unique(moving_apex, axis=0)
			pred_apex = moving_apex_u + np.dot(gaussian_kernel(moving, moving_apex_u), params[1])
			x_pa, y_pa, z_pa = get_unique_plot_points(pred_apex)
			moving_base_u = np.unique(moving_base, axis=0)
			pred_base = moving_base_u + np.dot(gaussian_kernel(moving, moving_base_u), params[1])
			x_pb, y_pb, z_pb = get_unique_plot_points(pred_base)

			fig = plt.figure()
			ax0 = fig.add_subplot(221, projection='3d')
			ax0.set_title('Fixed P & Tz')
			ax0.scatter(x_fptz, y_fptz, z_fptz, c='y', marker='.', alpha=0.2)
			ax0.scatter(x_fa, y_fa, z_fa, c='k', marker='^', alpha=1)
			ax0.scatter(x_fb, y_fb, z_fb, c='k', marker='v', alpha=1)

			ax1 = fig.add_subplot(222, projection='3d')
			ax1.set_title('Moving P & Tz')
			ax1.scatter(x_mptz, y_mptz, z_mptz, c='r', marker='.', alpha=0.2)
			ax1.scatter(x_ma, y_ma, z_ma, c='k', marker='^', alpha=1)
			ax1.scatter(x_mb, y_mb, z_mb, c='k', marker='v', alpha=1)

			ax2 = fig.add_subplot(223, projection='3d')
			ax2.set_title('Moving2Fixed P & Tz')
			ax2.scatter(x_pred, y_pred, z_pred, c='g', marker='.', alpha=0.2)
			ax2.scatter(x_pa, y_pa, z_pa, c='b', marker='^', alpha=1)
			ax2.scatter(x_pb, y_pb, z_pb, c='b', marker='v', alpha=1)

			ax3 = fig.add_subplot(224, projection='3d')
			ax3.set_title('Fixed Tz & Moving2Fixed Tz')
			ax3.scatter(x_ftz, y_ftz, z_ftz, c='m', marker='.', alpha=0.2)
			ax3.scatter(x_ptz, y_ptz, z_ptz, c='g', marker='.', alpha=0.2)

			set_plot_ax_lims([ax0, ax1, ax2, ax3])

			ref_data = patient_data[perm[0]][7]
			ref_data_mc = ref_data - np.mean(ref_data, axis=0)

			# Scale the data so we can compute metrics with correct values.
			pred_dn = denormalize(pred_u, ref_data_mc)
			fixed_dn = denormalize(fixed_u, ref_data_mc)
			d_c_p = round(chamfer(fixed_dn, pred_dn), 3)
			d_h_p = round(hausdorff_2way(fixed_dn, pred_dn), 3)

			pred_transition_zone_dn = denormalize(pred_transition_zone_u, ref_data_mc)
			fixed_transition_zone_dn = denormalize(fixed_transition_zone_u, ref_data_mc)
			d_c_tz = round(chamfer(fixed_transition_zone_dn, pred_transition_zone_dn), 3)
			d_h_tz = round(hausdorff_2way(fixed_transition_zone_dn, pred_transition_zone_dn), 3)

			fixed_apex_dn = denormalize(fixed_apex_u, ref_data_mc)
			fixed_base_dn = denormalize(fixed_base_u, ref_data_mc)
			pred_apex_dn = denormalize(pred_apex, ref_data_mc)
			pred_base_dn = denormalize(pred_base, ref_data_mc)
			d_apex = round(np.linalg.norm(fixed_apex_dn - pred_apex_dn), 3)
			d_base = round(np.linalg.norm(fixed_base_dn - pred_base_dn), 3)

			fig.suptitle(str(filenames[indxs[i]][0]) + ' ' + str(tags[perm[1]]) + ' to ' + str(tags[perm[0]]))
			plt.show()
			plt.savefig('./prostate_results/CPD-PTz2PTz-' + str(filenames[indxs[i]][0]) + '_' + str(tags[perm[1]]) + '-to-' + str(tags[perm[0]]) + '.png', dpi=300)
			plt.close()

			result_string = str(filenames[indxs[i]][0]) + '\t' + str(tags[perm[1]]) + '\t' + str(tags[perm[0]]) + '\t' + str(t) + '\t' + str(d_c_p) + '\t' + str(d_h_p) + '\t' + str(d_c_tz) + '\t' + str(d_h_tz) + '\t' + str(d_apex) + '\t' + str(d_base) + '\n'
			f = open('./prostate_results/CPD-PTz2PTz.txt', 'a')
			f.write(result_string)
			f.close()
	'''

predict_mr_us_file_Slices('no-reg-1')
