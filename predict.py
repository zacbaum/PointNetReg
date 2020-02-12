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
from keras.models import load_model
from keras.engine.topology import Layer
from cpd import deformable_registration, gaussian_kernel

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

def get_prostate_data(data, indxs):
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

				if prostate_j_data.shape[0] > dims[0]:
					prostate_j_data = prostate_j_data[np.random.randint(prostate_j_data.shape[0], size=dims[0]), :]
				prostate_j_data_centered = prostate_j_data - np.mean(mc_data, axis=0)
				prostate_j_data_normalized = 2 * (prostate_j_data_centered - np.min(norm_data)) / np.ptp(norm_data) - 1
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

def set_plot_ax_lims(axes, limit):
	for ax in axes:
		ax.set_xlim([-limit, limit])
		ax.set_ylim([-limit, limit])
		ax.set_zlim([-limit, limit])

def denormalize(unique_points, reference_points):
	unique_points_dn = 0.5 * ((unique_points * np.ptp(reference_points)) + (2 * np.min(reference_points)) + np.ptp(reference_points))
	return unique_points_dn

def get_summary_stats(names, data_lists):
	sum_stats_str = '\n\tMean\tMin\tMax\n'
	for i in range(len(names)):
		sum_stats_str += names[i] + ':\t' + str(round(sum(data_lists[i])/len(data_lists[i]), 3)) + '\t' + str(round(min(data_lists[i]), 3)) + '\t' + str(round(max(data_lists[i]), 3)) + '\n'
	return sum_stats_str

if not os.path.exists('./prostate_results/'):
	os.mkdir('./prostate_results/')

# Load the model.
model = load_model('CTN-DGX3.hdf5',
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
plot_ax_lim = 1

'''
d_c_list = []
d_h_list = []
d_apex_list = []
d_base_list = []
time_list = []
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
		t = time.time()
		pred = model.predict([[np.array(fixed_prostate)],
							  [np.array(moving_prostate)],
							  [np.array(moving_prostate)]])
		t = round(time.time() - t, 3)
		time_list.append(t)
		pred_u = np.unique(pred[0], axis=0)
		x_pred, y_pred, z_pred = get_unique_plot_points(pred[0])

		# Moving2Fixed Apex & Base
		pred_apex = model.predict([[np.array(fixed_prostate)],
								   [np.array(moving_prostate)],
								   [np.array(moving_apex)]])
		pred_apex_u = np.unique(pred_apex[0], axis=0)
		x_pa, y_pa, z_pa = get_unique_plot_points(pred_apex[0])
		pred_base = model.predict([[np.array(fixed_prostate)],
								   [np.array(moving_prostate)],
								   [np.array(moving_base)]])
		pred_base_u = np.unique(pred_base[0], axis=0)
		x_pb, y_pb, z_pb = get_unique_plot_points(pred_base[0])

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
		ax2.scatter(x_fp, y_fp, z_fp, c='y', marker='.', alpha=0.1)
		ax2.scatter(x_pred, y_pred, z_pred, c='g', marker='.', alpha=0.5)
		ax2.scatter(x_fa, y_fa, z_fa, c='k', marker='^', alpha=1)
		ax2.scatter(x_fb, y_fb, z_fb, c='k', marker='v', alpha=1)
		ax2.scatter(x_pa, y_pa, z_pa, c='b', marker='^', alpha=1)
		ax2.scatter(x_pb, y_pb, z_pb, c='b', marker='v', alpha=1)

		ax2.scatter(x_ma, y_ma, z_ma, c='m', marker='^', alpha=1)
		ax2.scatter(x_mb, y_mb, z_mb, c='m', marker='v', alpha=1)
		ax2.quiver(x_ma, y_ma, z_ma, [a_i - b_i for a_i, b_i in zip(x_pa, x_ma)], [a_i - b_i for a_i, b_i in zip(y_pa, y_ma)], [a_i - b_i for a_i, b_i in zip(z_pa, z_ma)], color='k')
		ax2.quiver(x_mb, y_mb, z_mb, [a_i - b_i for a_i, b_i in zip(x_pb, x_mb)], [a_i - b_i for a_i, b_i in zip(y_pb, y_mb)], [a_i - b_i for a_i, b_i in zip(z_pb, z_mb)], color='k')

		ax3 = fig.add_subplot(224, projection='3d')
		ax3.set_title('Fixed Tz & Moving2Fixed P')
		ax3.scatter(x_ftz, y_ftz, z_ftz, c='m', marker='.', alpha=0.2)
		ax3.scatter(x_pred, y_pred, z_pred, c='g', marker='.', alpha=0.2)
		ax3.scatter(x_pa, y_pa, z_pa, c='b', marker='^', alpha=1)
		ax3.scatter(x_pb, y_pb, z_pb, c='b', marker='v', alpha=1)

		set_plot_ax_lims([ax0, ax1, ax2, ax3], plot_ax_lim)

		ref_data = patient_data[perm[0]][7]
		ref_data_mc = ref_data - np.mean(ref_data, axis=0)

		# Scale the data so we can compute metrics with correct values.
		pred_dn = denormalize(pred_u, ref_data_mc)
		fixed_prostate_dn = denormalize(fixed_prostate_u, ref_data_mc)
		d_c = round(chamfer(fixed_prostate_dn, pred_dn), 3)
		d_c_list.append(d_c)
		d_h = round(hausdorff_2way(fixed_prostate_dn, pred_dn), 3)
		d_h_list.append(d_h)

		fixed_apex_dn = denormalize(fixed_apex_u, ref_data_mc)
		fixed_base_dn = denormalize(fixed_base_u, ref_data_mc)
		pred_apex_dn = denormalize(pred_apex_u, ref_data_mc)
		pred_base_dn = denormalize(pred_base_u, ref_data_mc)
		d_apex = round(np.linalg.norm(fixed_apex_dn - pred_apex_dn), 3)
		d_apex_list.append(d_apex)
		d_base = round(np.linalg.norm(fixed_base_dn - pred_base_dn), 3)
		d_base_list.append(d_base)

		fig.suptitle(str(filenames[indxs[i]][0]) + ' ' + str(tags[perm[1]]) + ' to ' + str(tags[perm[0]]) + ' (CD: ' + str('%.2f') % d_c + ', HD: ' + str('%.2f') % d_h + ', Apex: ' + str('%.2f') % d_apex + ', Base: ' + str('%.2f') % d_base + ')')
		plt.show()
		plt.savefig('./prostate_results/P2P-' + str(filenames[indxs[i]][0]) + '_' + str(tags[perm[1]]) + '-to-' + str(tags[perm[0]]) + '.png', dpi=300)
		plt.close()

		result_string = str(filenames[indxs[i]][0]) + '\t' + str(tags[perm[1]]) + '\t' + str(tags[perm[0]]) + '\t' + str(t) + '\t' + str(d_c) + '\t' + str(d_h) + '\t' + str(d_apex) + '\t' + str(d_base) + '\n'
		f = open('./prostate_results/P2P.txt', 'a')
		f.write(result_string)
		f.close()

print(get_summary_stats(['Time', 'CD', 'HD', 'Apex', 'Base'], [time_list, d_c_list, d_h_list, d_apex_list, d_base_list]))

d_c_list = []
d_h_list = []
d_apex_list = []
d_base_list = []
time_list = []
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
		time_list.append(t)
		pred_u = np.unique(pred[0], axis=0)
		x_pred, y_pred, z_pred = get_unique_plot_points(pred[0])

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
		ax2.scatter(x_pred, y_pred, z_pred, c='g', marker='.', alpha=0.2)
		ax2.scatter(x_pa, y_pa, z_pa, c='b', marker='^', alpha=1)
		ax2.scatter(x_pb, y_pb, z_pb, c='b', marker='v', alpha=1)

		ax3 = fig.add_subplot(224, projection='3d')
		ax3.set_title('Fixed P & Tz & Moving2Fixed P & Tz')
		ax3.scatter(x_fptz, y_fptz, z_fptz, c='y', marker='.', alpha=0.2)
		ax3.scatter(x_pred, y_pred, z_pred, c='g', marker='.', alpha=0.2)
		ax3.scatter(x_fa, y_fa, z_fa, c='k', marker='^', alpha=1)
		ax3.scatter(x_fb, y_fb, z_fb, c='k', marker='v', alpha=1)
		ax3.scatter(x_pa, y_pa, z_pa, c='b', marker='^', alpha=1)
		ax3.scatter(x_pb, y_pb, z_pb, c='b', marker='v', alpha=1)

		set_plot_ax_lims([ax0, ax1, ax2, ax3], plot_ax_lim)

		ref_data = patient_data[perm[0]][7]
		ref_data_mc = ref_data - np.mean(ref_data, axis=0)

		# Scale the data so we can compute metrics with correct values.
		pred_dn = denormalize(pred_u, ref_data_mc)
		fixed_dn = denormalize(fixed_u, ref_data_mc)
		d_c = round(chamfer(fixed_dn, pred_dn), 3)
		d_c_list.append(d_c)
		d_h = round(hausdorff_2way(fixed_dn, pred_dn), 3)
		d_h_list.append(d_h)

		fixed_apex_dn = denormalize(fixed_apex_u, ref_data_mc)
		fixed_base_dn = denormalize(fixed_base_u, ref_data_mc)
		pred_apex_dn = denormalize(pred_apex_u, ref_data_mc)
		pred_base_dn = denormalize(pred_base_u, ref_data_mc)
		d_apex = round(np.linalg.norm(fixed_apex_dn - pred_apex_dn), 3)
		d_apex_list.append(d_apex)
		d_base = round(np.linalg.norm(fixed_base_dn - pred_base_dn), 3)
		d_base_list.append(d_base)

		fig.suptitle(str(filenames[indxs[i]][0]) + ' ' + str(tags[perm[1]]) + ' to ' + str(tags[perm[0]]) + ' (CD: ' + str('%.2f') % d_c + ', HD: ' + str('%.2f') % d_h + ', Apex: ' + str('%.2f') % d_apex + ', Base: ' + str('%.2f') % d_base + ')')
		plt.show()
		plt.savefig('./prostate_results/PTz2PTz-' + str(filenames[indxs[i]][0]) + '_' + str(tags[perm[1]]) + '-to-' + str(tags[perm[0]]) + '.png', dpi=300)
		plt.close()

		result_string = str(filenames[indxs[i]][0]) + '\t' + str(tags[perm[1]]) + '\t' + str(tags[perm[0]]) + '\t' + str(t) + '\t' + str(d_c) + '\t' + str(d_h) + '\t' + str(d_apex) + '\t' + str(d_base) + '\n'
		f = open('./prostate_results/PTz2PTz.txt', 'a')
		f.write(result_string)
		f.close()

print(get_summary_stats(['Time', 'CD', 'HD', 'Apex', 'Base'], [time_list, d_c_list, d_h_list, d_apex_list, d_base_list]))
'''
d_c_list = []
d_h_list = []
d_apex_list = []
d_base_list = []
time_list = []
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
		time_list.append(t)
		pred_u = np.unique(pred, axis=0)
		x_pred, y_pred, z_pred = get_unique_plot_points(pred)

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
		ax2.scatter(x_fp, y_fp, z_fp, c='y', marker='.', alpha=0.1)
		ax2.scatter(x_pred, y_pred, z_pred, c='g', marker='.', alpha=0.1)
		ax2.scatter(x_fa, y_fa, z_fa, c='k', marker='^', alpha=1)
		ax2.scatter(x_fb, y_fb, z_fb, c='k', marker='v', alpha=1)
		ax2.scatter(x_pa, y_pa, z_pa, c='b', marker='^', alpha=1)
		ax2.scatter(x_pb, y_pb, z_pb, c='b', marker='v', alpha=1)

		ax3 = fig.add_subplot(224, projection='3d')
		ax3.set_title('Fixed Tz & Moving2Fixed P')
		ax3.scatter(x_ftz, y_ftz, z_ftz, c='m', marker='.', alpha=0.2)
		ax3.scatter(x_pred, y_pred, z_pred, c='g', marker='.', alpha=0.2)
		ax3.scatter(x_pa, y_pa, z_pa, c='b', marker='^', alpha=1)
		ax3.scatter(x_pb, y_pb, z_pb, c='b', marker='v', alpha=1)

		set_plot_ax_lims([ax0, ax1, ax2, ax3], plot_ax_lim)

		ref_data = patient_data[perm[0]][7]
		ref_data_mc = ref_data - np.mean(ref_data, axis=0)

		# Scale the data so we can compute metrics with correct values.
		pred_dn = denormalize(pred_u, ref_data_mc)
		fixed_prostate_dn = denormalize(fixed_prostate_u, ref_data_mc)
		d_c = round(chamfer(fixed_prostate_dn, pred_dn), 3)
		d_c_list.append(d_c)
		d_h = round(hausdorff_2way(fixed_prostate_dn, pred_dn), 3)
		d_h_list.append(d_h)

		fixed_apex_dn = denormalize(fixed_apex_u, ref_data_mc)
		fixed_base_dn = denormalize(fixed_base_u, ref_data_mc)
		pred_apex_dn = denormalize(pred_apex, ref_data_mc)
		pred_base_dn = denormalize(pred_base, ref_data_mc)
		d_apex = round(np.linalg.norm(fixed_apex_dn - pred_apex_dn), 3)
		d_apex_list.append(d_apex)
		d_base = round(np.linalg.norm(fixed_base_dn - pred_base_dn), 3)
		d_base_list.append(d_base)

		fig.suptitle(str(filenames[indxs[i]][0]) + ' ' + str(tags[perm[1]]) + ' to ' + str(tags[perm[0]]) + ' (CD: ' + str('%.2f') % d_c + ', HD: ' + str('%.2f') % d_h + ', Apex: ' + str('%.2f') % d_apex + ', Base: ' + str('%.2f') % d_base + ')')
		plt.show()
		plt.savefig('./prostate_results/CPD-P2P-' + str(filenames[indxs[i]][0]) + '_' + str(tags[perm[1]]) + '-to-' + str(tags[perm[0]]) + '.png', dpi=300)
		plt.close()

		result_string = str(filenames[indxs[i]][0]) + '\t' + str(tags[perm[1]]) + '\t' + str(tags[perm[0]]) + '\t' + str(t) + '\t' + str(d_c) + '\t' + str(d_h) + '\t' + str(d_apex) + '\t' + str(d_base) + '\n'
		f = open('./prostate_results/CPD-P2P.txt', 'a')
		f.write(result_string)
		f.close()

print(get_summary_stats(['Time', 'CD', 'HD', 'Apex', 'Base'], [time_list, d_c_list, d_h_list, d_apex_list, d_base_list]))

d_c_list = []
d_h_list = []
d_apex_list = []
d_base_list = []
time_list = []
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
		time_list.append(t)
		pred_u = np.unique(pred, axis=0)
		x_pred, y_pred, z_pred = get_unique_plot_points(pred)

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
		ax3.set_title('Fixed P & Tz & Moving2Fixed P & Tz')
		ax3.scatter(x_fptz, y_fptz, z_fptz, c='y', marker='.', alpha=0.2)
		ax3.scatter(x_pred, y_pred, z_pred, c='g', marker='.', alpha=0.2)
		ax3.scatter(x_fa, y_fa, z_fa, c='k', marker='^', alpha=1)
		ax3.scatter(x_fb, y_fb, z_fb, c='k', marker='v', alpha=1)
		ax3.scatter(x_pa, y_pa, z_pa, c='b', marker='^', alpha=1)
		ax3.scatter(x_pb, y_pb, z_pb, c='b', marker='v', alpha=1)

		set_plot_ax_lims([ax0, ax1, ax2, ax3], plot_ax_lim)

		ref_data = patient_data[perm[0]][7]
		ref_data_mc = ref_data - np.mean(ref_data, axis=0)

		# Scale the data so we can compute metrics with correct values.
		pred_dn = denormalize(pred_u, ref_data_mc)
		fixed_dn = denormalize(fixed_u, ref_data_mc)
		d_c = round(chamfer(fixed_dn, pred_dn), 3)
		d_c_list.append(d_c)
		d_h = round(hausdorff_2way(fixed_dn, pred_dn), 3)
		d_h_list.append(d_h)

		fixed_apex_dn = denormalize(fixed_apex_u, ref_data_mc)
		fixed_base_dn = denormalize(fixed_base_u, ref_data_mc)
		pred_apex_dn = denormalize(pred_apex, ref_data_mc)
		pred_base_dn = denormalize(pred_base, ref_data_mc)
		d_apex = round(np.linalg.norm(fixed_apex_dn - pred_apex_dn), 3)
		d_apex_list.append(d_apex)
		d_base = round(np.linalg.norm(fixed_base_dn - pred_base_dn), 3)
		d_base_list.append(d_base)

		fig.suptitle(str(filenames[indxs[i]][0]) + ' ' + str(tags[perm[1]]) + ' to ' + str(tags[perm[0]]) + ' (CD: ' + str('%.2f') % d_c + ', HD: ' + str('%.2f') % d_h + ', Apex: ' + str('%.2f') % d_apex + ', Base: ' + str('%.2f') % d_base + ')')
		plt.show()
		plt.savefig('./prostate_results/CPD-PTz2PTz-' + str(filenames[indxs[i]][0]) + '_' + str(tags[perm[1]]) + '-to-' + str(tags[perm[0]]) + '.png', dpi=300)
		plt.close()

		result_string = str(filenames[indxs[i]][0]) + '\t' + str(tags[perm[1]]) + '\t' + str(tags[perm[0]]) + '\t' + str(t) + '\t' + str(d_c) + '\t' + str(d_h) + '\t' + str(d_apex) + '\t' + str(d_base) + '\n'
		f = open('./prostate_results/CPD-PTz2PTz.txt', 'a')
		f.write(result_string)
		f.close()

print(get_summary_stats(['Time', 'CD', 'HD', 'Apex', 'Base'], [time_list, d_c_list, d_h_list, d_apex_list, d_base_list]))