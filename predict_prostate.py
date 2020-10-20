import tensorflow as tf
from keras import backend as K
import numpy as np
import scipy
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
matplotlib.use('AGG')
import os
import re
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import time
from cpd import deformable_registration, gaussian_kernel
from model import MatMul, FreePointTransformer, TPSTransformNet
import h5py
import cv2
from tensorflow.keras.models import load_model

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

def point_plane_distance(P, a, b, c, d):
    return np.abs(P[0] * a + P[1] * b + P[2] * c + d) / np.sqrt(a*a + b*b + c*c)

def get_unique_plot_points(points):
    points_u = np.unique(points, axis=0)
    x = [i[0] for i in points_u]
    y = [i[1] for i in points_u]
    z = [i[2] for i in points_u]
    return [x, y, z]

def add_1s_dim(pts):
    pts_with_ones = np.ones((pts.shape[0], pts.shape[1] + 1))
    pts_with_ones[:, :-1] = pts
    return pts_with_ones

def get_mr_us_data(fixed_fname, moving_fname, dense=False):
    
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
        print(case)
        current = []
        current_prostate_metrics = []
        for mask in range(num_labels[case]):
            
            fixed_points = np.array(fixed[fixed_keys[indx]])
            fixed_contour_points = []
            for z in range(fixed_points.shape[0]):
                if not dense:
                    edged = cv2.Canny(fixed_points[z], 0, 1)
                    indices = np.where(edged != [0])
                else:
                    indices = np.where(fixed_points[z] != 0)
                coordinates = zip(indices[0], indices[1])
                for coordinate in coordinates:
                    if dense and coordinate[0] % 2 == 1 and coordinate[1] % 2 == 1:
                        fixed_contour_points.append([pxl_to_mm_scale * coordinate[0], pxl_to_mm_scale * coordinate[1], pxl_to_mm_scale * z])
                    elif not dense:
                        fixed_contour_points.append([pxl_to_mm_scale * coordinate[0], pxl_to_mm_scale * coordinate[1], pxl_to_mm_scale * z])            
            fixed_contour_points = np.array(fixed_contour_points)
            
            moving_points = np.array(moving[moving_keys[indx]])
            moving_contour_points = []
            for z in range(moving_points.shape[0]):
                if not dense:
                    edged = cv2.Canny(moving_points[z], 0, 1)
                    indices = np.where(edged != [0])
                else:
                    indices = np.where(moving_points[z] != 0)
                coordinates = zip(indices[0], indices[1])
                for coordinate in coordinates:
                    if dense and coordinate[0] % 2 == 1 and coordinate[1] % 2 == 1:
                        moving_contour_points.append([pxl_to_mm_scale * coordinate[0], pxl_to_mm_scale * coordinate[1], pxl_to_mm_scale * z])
                    elif not dense:
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

def predict_mr_us_file(name, path, dims=[2**11, 3]):
    if not os.path.exists('./prostate_results-' + name + '/'):
            os.mkdir('./prostate_results-' + name + '/')

    header_string = 'P_ID\tTIME\tDC_P\tDH_P\tDC_R1\tDH_R1\tTRE_R1\tDC_R2\tDH_R2\tTRE_R2\tDC_R3\tDH_R3\tTRE_R3\tDC_R4\tDH_R4\tTRE_R4\tDC_R5\tDH_R5\tTRE_R5\n'
    f = open('./prostate_results-' + name + '/P2P.txt', 'a')
    f.write(header_string)
    f.close()
    '''
    f = open('./prostate_results-' + name + '/CPD-P2P.txt', 'a')
    f.write(header_string)
    f.close()
    '''

    # Load the model.
    if 'baseline' in name:
        model = TPSTransformNet(
            dims[0],
            dims=4,
            tps_features=27,
            sigma=1.0,
        )
    if 'fpt' in name:
        model = FreePointTransformer(
            dims[0],
            dims=4,
            skips=False,
        )
    model.load_weights(path + '.h5')

    if not os.path.exists('./mrus/prostates_volumes.npy') or not os.path.exists('./mrus/prostate_metrics_volumes.npy'):
        all_prostates, metrics = get_mr_us_data('./mrus/us_labels_resampled800_post3.h5', './mrus/mr_labels_resampled800_post3.h5', dense=True)
        np.save('./mrus/prostates_volumes.npy', all_prostates)
        np.save('./mrus/prostate_metrics_volumes.npy', metrics)
    else:
        all_prostates = np.load('./mrus/prostates_volumes.npy', allow_pickle=True)
        metrics = np.load('./mrus/prostate_metrics_volumes.npy', allow_pickle=True)

    max_iters = len(all_prostates)
    split = 0.7

    # CTN - Contour to Contour
    for i in range(int(split * max_iters), max_iters):

        # Fixed Prostate
        fixed_prostate = all_prostates[i][0][0]
        if fixed_prostate.shape[0] > dims[0]:
            fixed_prostate = fixed_prostate[np.random.choice(fixed_prostate.shape[0], size=dims[0], replace=False), :]
        else:
            fixed_prostate = np.resize(fixed_prostate, dims)
        fixed_prostate_u = np.unique(fixed_prostate, axis=0)
        fixed_prostate = add_1s_dim(fixed_prostate)
        # Fixed ROIs
        ROIs = [x for x in all_prostates[i][1:]]
        fixed_ROIs = [x[0] for x in ROIs]
        for r in range(len(fixed_ROIs)):
            if fixed_ROIs[r].shape[0] > dims[0]:
                fixed_ROIs[r] = fixed_ROIs[r][np.random.choice(r.shape[0], size=dims[0], replace=False), :]
            else:
                fixed_ROIs[r] = np.resize(fixed_ROIs[r], dims)
        fixed_ROIs_u = [np.unique(x[0], axis=0) for x in ROIs]
        fixed_ROIs = [add_1s_dim(x) for x in fixed_ROIs]

        # Moving Prostate
        moving_prostate = all_prostates[i][0][1]
        if moving_prostate.shape[0] > dims[0]:
            moving_prostate = moving_prostate[np.random.choice(moving_prostate.shape[0], size=dims[0], replace=False), :]
        else:
            moving_prostate = np.resize(moving_prostate, dims)
        moving_prostate_u = np.unique(moving_prostate, axis=0)
        moving_prostate = add_1s_dim(moving_prostate)
        # Moving ROIs
        ROIs = [x for x in all_prostates[i][1:]]
        moving_ROIs = [x[1] for x in ROIs]
        for r in range(len(moving_ROIs)):
            if moving_ROIs[r].shape[0] > dims[0]:
                moving_ROIs[r] = moving_ROIs[r][np.random.choice(r.shape[0], size=dims[0], replace=False), :]
            else:
                moving_ROIs[r] = np.resize(moving_ROIs[r], dims)
        moving_ROIs_u = [np.unique(x[1], axis=0) for x in ROIs]
        moving_ROIs = [add_1s_dim(x) for x in moving_ROIs]


        # Moving2Fixed Prostate
        t = time.time()
        #pred = moving_prostate
        pred = model.predict([[np.array(fixed_prostate)],
                              [np.array(moving_prostate)],
                              [np.array(moving_prostate)]])
        t = round(time.time() - t, 3)
        #pred_u = np.unique(pred, axis=0)
        pred_u = np.unique(pred[0], axis=0)[:, :-1]
        
        # Moving2Fixed ROIs
        #pred_ROIs = moving_ROIs
        pred_ROIs = [model.predict([[np.array(fixed_prostate)],
                                    [np.array(moving_prostate)],
                                    [np.array(x)]]) for x in moving_ROIs]
        #pred_ROIs_u = [np.unique(x, axis=0) for x in pred_ROIs]
        pred_ROIs_u = [np.unique(x[0], axis=0)[:, :-1] for x in pred_ROIs]
        
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
        ax0.scatter(x_fp, y_fp, z_fp, c='y', marker='.', alpha=0.1)
        
        ax1 = fig.add_subplot(222, projection='3d')
        ax1.set_title('Moving')
        ax1.scatter(x_mp, y_mp, z_mp, c='r', marker='.', alpha=0.1)
    
        ax2 = fig.add_subplot(223, projection='3d')
        ax2.set_title('Registered Contours')
        #ax2.scatter(x_fp, y_fp, z_fp, c='y', marker='.', alpha=0.2)
        ax2.scatter(x_pred, y_pred, z_pred, c='g', marker='.', alpha=0.1)
    
        ax3 = fig.add_subplot(224, projection='3d')
        ax3.set_title('Registered ROIs')
        for roi in fixed_ROIs_xyz:
            ax3.scatter(roi[0], roi[1], roi[2], c='y', marker='.', alpha=0.1)
        for roi in pred_ROIs_xyz:
            ax3.scatter(roi[0], roi[1], roi[2], c='g', marker='.', alpha=0.1)

        set_plot_ax_lims([ax0, ax1, ax2, ax3], limit=30)
        fig.suptitle('Patient ' + str(i + 1) + ' - MR to US')
        plt.show()

        def rotate(angle):
            ax0.view_init(azim=angle)
            ax1.view_init(azim=angle)
            ax2.view_init(azim=angle)
            ax3.view_init(azim=angle)

        angle = 3
        ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=100)
        ani.save(
            './prostate_results-' + name + '/P2P-' + str(i + 1) + '.gif',
            writer=animation.PillowWriter(fps=20),
        )

        ROIs_string = ''
        for roi in range(5):
            d_c = str(d_c_ROIs[roi]) if roi < len(d_c_ROIs) else ' '
            d_h = str(d_h_ROIs[roi]) if roi < len(d_h_ROIs) else ' '
            d_RE = str(d_RE_ROIs[roi]) if roi < len(d_RE_ROIs) else ' '
            ROIs_string += str(d_c) + '\t' + str(d_h) + '\t' + str(d_RE) + '\t'
        result_string = str(i + 1) + '\t' + str(t) + '\t' + \
                            str(d_c_p) + '\t' + str(d_h_p) + '\t' + \
                            ROIs_string + '\n'
        f = open('./prostate_results-' + name + '/P2P.txt', 'a')
        f.write(result_string)
        f.close()

        #plt.savefig('./prostate_results-' + name + '/P2P-' + str(i + 1) + '.png', dpi=300)
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
        f = open('./prostate_results-' + name + '/CPD-P2P.txt', 'a')
        f.write(result_string)
        f.close()

        fig.suptitle('Patient ' + str(i + 1) + ' - MR to US')
        plt.show()
        plt.savefig('./prostate_results-' + name + '/CPD-P2P-' + str(i + 1) + '.png', dpi=300)
        plt.close()
    '''

def predict_mr_us_file_Slices(name, path):
    if not os.path.exists('./prostate_results-' + name + '/'):
            os.mkdir('./prostate_results-' + name + '/')

    #header_string = 'P_ID\tTIME\tDC_P\tDH_P\tDC_R1\tDH_R1\tTRE_R1\tDC_R2\tDH_R2\tTRE_R2\tDC_R3\tDH_R3\tTRE_R3\tDC_R4\tDH_R4\tTRE_R4\tDC_R5\tDH_R5\tTRE_R5\n'
    #f = open('./prostate_results-' + name + '/P2P.txt', 'a')
    #f.write(header_string)
    #f.close()
    #f = open('./prostate_results-' + name + '/CPD-P2P.txt', 'a')
    #f.write(header_string)
    #f.close()
    # Load the model.
    #model = FreePointTransformer(2048)
    #model.load_weights(path + '.h5')

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
        #                     [np.array(moving_prostate)],
        #                     [np.array(moving_prostate)]])
        t = round(time.time() - t, 3)
        pred_u = np.unique(pred, axis=0)
        #pred_u = np.unique(pred[0], axis=0)
        
        # Moving2Fixed ROIs
        pred_ROIs = moving_ROIs
        #pred_ROIs = [model.predict([[np.array(fixed_prostate_slices)],
        #                           [np.array(moving_prostate)],
        #                           [np.array(x)]]) for x in moving_ROIs]
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
        f = open('./prostate_results-' + name + '/P2P.txt', 'a')
        f.write(result_string)
        f.close()
        '''

        fig.suptitle('Patient ' + str(i + 1) + ' - MR to US')
        plt.show()
        plt.savefig('./prostate_results-' + name + '/P2P-' + str(i + 1) + '.png', dpi=300)
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
        f = open('./prostate_results-' + name + '/CPD-P2P.txt', 'a')
        f.write(result_string)
        f.close()

        fig.suptitle('Patient ' + str(i + 1) + ' - MR to US')
        plt.show()
        plt.savefig('./prostate_results-' + name + '/CPD-P2P-' + str(i + 1) + '.png', dpi=300)
        plt.close()
    '''

def predict_mr_us_file_Sweep(name, path, sweeps):
    if not os.path.exists('./prostate_results-' + name + '/'):
            os.mkdir('./prostate_results-' + name + '/')

    header_string = 'P_ID\tTIME\tDC_P\tDH_P\tDC_R1\tDH_R1\tTRE_R1\tDC_R2\tDH_R2\tTRE_R2\tDC_R3\tDH_R3\tTRE_R3\tDC_R4\tDH_R4\tTRE_R4\tDC_R5\tDH_R5\tTRE_R5\n'
    f = open('./prostate_results-' + name + '/P2P.txt', 'a')
    f.write(header_string)
    f.close()
    '''
    f = open('./prostate_results-' + name + '/CPD-P2P.txt', 'a')
    f.write(header_string)
    f.close()
    '''
    # Load the model.
    if 'baseline' in name:
        model = TPSTransformNet(
            2048,
            dims=4,
            tps_features=27,
            sigma=1.0,
        )
    if 'fpt' in name:
        model = FreePointTransformer(
            2048,
            dims=4,
            skips=False,
        )
    model.load_weights(path + '.h5')

    dims = [2048, 3]
    if not os.path.exists('./mrus/prostates.npy') or not os.path.exists('./mrus/prostate_metrics.npy'):
        all_prostates, metrics = get_mr_us_data('./mrus/us_labels_resampled800_post3.h5', './mrus/mr_labels_resampled800_post3.h5')
        np.save('./mrus/prostates.npy', all_prostates)
        np.save('./mrus/prostate_metrics.npy', metrics)
    else:
        all_prostates = np.load('./mrus/prostates.npy', allow_pickle=True)
        metrics = np.load('./mrus/prostate_metrics.npy', allow_pickle=True)

    max_iters = len(all_prostates)
    split = 0.7

    # CTN - Contour to Contour
    for i in range(int(split * max_iters), max_iters):

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        thresh = 0.02
        fixed_prostate = all_prostates[i][0][0]
        swept_prostate = np.array([])

        TRUS_tip = np.array([0, 1, -1])
        TRUS_end = np.array([0, -1, -1])

        if sweeps == 1:
            P = np.array([0, 0, 1])
            a, b, c = np.cross(TRUS_tip - TRUS_end, TRUS_tip - P)
            d = -(a * P[0] + b * P[1] + c * P[2])
            temp = np.array([point for point in fixed_prostate if point_plane_distance(point, a, b, c, d) < thresh])
            swept_prostate = np.concatenate((swept_prostate, temp), axis=0) if swept_prostate.size else temp

        elif sweeps == 2:
            P = np.array([-0.5, 0, 1])
            a, b, c = np.cross(TRUS_tip - TRUS_end, TRUS_tip - P)
            d = -(a * P[0] + b * P[1] + c * P[2])
            temp = np.array([point for point in fixed_prostate if point_plane_distance(point, a, b, c, d) < thresh])
            swept_prostate = np.concatenate((swept_prostate, temp), axis=0) if swept_prostate.size else temp
            
            P = np.array([0.5, 0, 1])
            a, b, c = np.cross(TRUS_tip - TRUS_end, TRUS_tip - P)
            d = -(a * P[0] + b * P[1] + c * P[2])
            temp = np.array([point for point in fixed_prostate if point_plane_distance(point, a, b, c, d) < thresh])
            swept_prostate = np.concatenate((swept_prostate, temp), axis=0) if swept_prostate.size else temp

        else:
            end_points = np.linspace(-1, 1, sweeps)
            for end_point in end_points:
                P = np.array([end_point, 0, 1])
                a, b, c = np.cross(TRUS_tip - TRUS_end, TRUS_tip - P)
                d = -(a * P[0] + b * P[1] + c * P[2])
                temp = np.array([point for point in fixed_prostate if point_plane_distance(point, a, b, c, d) < thresh])
                swept_prostate = np.concatenate((swept_prostate, temp), axis=0) if swept_prostate.size else temp
        
        swept_prostate = np.reshape(swept_prostate, (-1, dims[1]))

        # Fixed Prostate
        if swept_prostate.shape[0] > dims[0]:
            swept_prostate = swept_prostate[np.random.choice(swept_prostate.shape[0], size=dims[0], replace=False), :]
        else:
            swept_prostate = np.resize(swept_prostate, dims)
        swept_prostate = add_1s_dim(swept_prostate)
        if fixed_prostate.shape[0] > dims[0]:
            fixed_prostate = fixed_prostate[np.random.choice(fixed_prostate.shape[0], size=dims[0], replace=False), :]
        else:
            fixed_prostate = np.resize(fixed_prostate, dims)
        fixed_prostate_u = np.unique(fixed_prostate, axis=0)
        fixed_prostate = add_1s_dim(fixed_prostate)
        # Fixed ROIs
        ROIs = [x for x in all_prostates[i][1:]]
        fixed_ROIs = [x[0] for x in ROIs]
        for r in range(len(fixed_ROIs)):
            if fixed_ROIs[r].shape[0] > dims[0]:
                fixed_ROIs[r] = fixed_ROIs[r][np.random.choice(r.shape[0], size=dims[0], replace=False), :]
            else:
                fixed_ROIs[r] = np.resize(fixed_ROIs[r], dims)
        fixed_ROIs_u = [np.unique(x[0], axis=0) for x in ROIs]
        fixed_ROIs = [add_1s_dim(x) for x in fixed_ROIs]

        # Moving Prostate
        moving_prostate = all_prostates[i][0][1]
        if moving_prostate.shape[0] > dims[0]:
            moving_prostate = moving_prostate[np.random.choice(moving_prostate.shape[0], size=dims[0], replace=False), :]
        else:
            moving_prostate = np.resize(moving_prostate, dims)
        moving_prostate_u = np.unique(moving_prostate, axis=0)
        moving_prostate = add_1s_dim(moving_prostate)
        # Moving ROIs
        ROIs = [x for x in all_prostates[i][1:]]
        moving_ROIs = [x[1] for x in ROIs]
        for r in range(len(moving_ROIs)):
            if moving_ROIs[r].shape[0] > dims[0]:
                moving_ROIs[r] = moving_ROIs[r][np.random.choice(r.shape[0], size=dims[0], replace=False), :]
            else:
                moving_ROIs[r] = np.resize(moving_ROIs[r], dims)
        moving_ROIs_u = [np.unique(x[1], axis=0) for x in ROIs]
        moving_ROIs = [add_1s_dim(x) for x in moving_ROIs]

        # Moving2Fixed Prostate
        t = time.time()
        #pred = moving_prostate
        pred = model.predict([[np.array(swept_prostate)],
                              [np.array(moving_prostate)],
                              [np.array(moving_prostate)]])
        t = round(time.time() - t, 3)
        #pred_u = np.unique(pred, axis=0)
        pred_u = np.unique(pred[0], axis=0)[:, :-1]
        
        # Moving2Fixed ROIs
        #pred_ROIs = moving_ROIs
        pred_ROIs = [model.predict([[np.array(swept_prostate)],
                                    [np.array(moving_prostate)],
                                    [np.array(x)]]) for x in moving_ROIs]
        #pred_ROIs_u = [np.unique(x, axis=0) for x in pred_ROIs]
        pred_ROIs_u = [np.unique(x[0], axis=0)[:, :-1] for x in pred_ROIs]
        
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

        '''
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
        f = open('./prostate_results-' + name + '/P2P.txt', 'a')
        f.write(result_string)
        f.close()

        '''
        fig.suptitle('Patient ' + str(i + 1) + ' - MR to US')
        plt.show()
        plt.savefig('./prostate_results-' + name + '/P2P-' + str(i + 1) + '.png', dpi=300)
        plt.close()
        '''
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
        f = open('./prostate_results-' + name + '/CPD-P2P.txt', 'a')
        f.write(result_string)
        f.close()

        fig.suptitle('Patient ' + str(i + 1) + ' - MR to US')
        plt.show()
        plt.savefig('./prostate_results-' + name + '/CPD-P2P-' + str(i + 1) + '.png', dpi=300)
        plt.close()
    '''

if __name__ == "__main__":

    #predict_mr_us_file("f2f_70-30_8192_baseline-cd_disp-rot-1e-5", "./models/full-to-full-volumes/8192/70-30/baseline-cd/disp-rot-1e-5/model", [2**13, 3])
    #predict_mr_us_file("f2f_70-30_8192_baseline-gmm_disp-rot-1e-5", "./models/full-to-full-volumes/8192/70-30/baseline-gmm/disp-rot-1e-5/model", [2**13, 3])
    #predict_mr_us_file("f2f_70-30_8192_fpt-cd_disp-rot-1e-5", "./models/full-to-full-volumes/8192/70-30/fpt-cd/disp-rot-1e-5/model", [2**13, 3])
    #predict_mr_us_file("f2f_70-30_8192_fpt-gmm_disp-rot-1e-5", "./models/full-to-full-volumes/8192/70-30/fpt-gmm/disp-rot-1e-5/model", [2**13, 3])

    predict_mr_us_file("f2f_70-30_4096_baseline-cd_disp-rot-1e-5", "./models/full-to-full-volumes/4096/70-30/baseline-cd/disp-rot/model", [2**12, 3])
    predict_mr_us_file("f2f_70-30_4096_baseline-gmm_disp-rot-1e-5", "./models/full-to-full-volumes/4096/70-30/baseline-gmm/disp-rot/model", [2**12, 3])
    predict_mr_us_file("f2f_70-30_4096_fpt-cd_disp-rot-1e-5", "./models/full-to-full-volumes/4096/70-30/fpt-cd/disp-rot/model", [2**12, 3])
    predict_mr_us_file("f2f_70-30_4096_fpt-gmm_disp-rot-1e-5", "./models/full-to-full-volumes/4096/70-30/fpt-gmm/disp-rot/model", [2**12, 3])

    predict_mr_us_file("f2f_70-30_2048_baseline-cd_disp-rot-1e-5", "./models/full-to-full-volumes/2048/70-30/baseline-cd/disp-rot/model")
    predict_mr_us_file("f2f_70-30_2048_baseline-gmm_disp-rot-1e-5", "./models/full-to-full-volumes/2048/70-30/baseline-gmm/disp-rot/model")
    predict_mr_us_file("f2f_70-30_2048_fpt-cd_disp-rot-1e-5", "./models/full-to-full-volumes/2048/70-30/fpt-cd/disp-rot/model")
    predict_mr_us_file("f2f_70-30_2048_fpt-gmm_disp-rot-1e-5", "./models/full-to-full-volumes/2048/70-30/fpt-gmm/disp-rot/model")

    '''
    # 70-30
    predict_mr_us_file("f2f_70-30_baseline-cd_disp-rot-1e-5-best", "./models/full-to-full-volumes/70-30/baseline-cd/disp-rot-1e-5/model-best")
    predict_mr_us_file("f2f_70-30_baseline-cd_disp-rot-1e-6-best", "./models/full-to-full-volumes/70-30/baseline-cd/disp-rot-1e-6/model-best")
    predict_mr_us_file("f2f_70-30_baseline-cd_disp-rot-1e-5", "./models/full-to-full-volumes/70-30/baseline-cd/disp-rot-1e-5/model")
    predict_mr_us_file("f2f_70-30_baseline-cd_disp-rot-1e-6", "./models/full-to-full-volumes/70-30/baseline-cd/disp-rot-1e-6/model")

    predict_mr_us_file("f2f_70-30_baseline-gmm_disp-rot-1e-5-best", "./models/full-to-full-volumes/70-30/baseline-gmm/disp-rot-1e-5/model-best")
    predict_mr_us_file("f2f_70-30_baseline-gmm_disp-rot-1e-6-best", "./models/full-to-full-volumes/70-30/baseline-gmm/disp-rot-1e-6/model-best")
    predict_mr_us_file("f2f_70-30_baseline-gmm_disp-rot-1e-5", "./models/full-to-full-volumes/70-30/baseline-gmm/disp-rot-1e-5/model")
    predict_mr_us_file("f2f_70-30_baseline-gmm_disp-rot-1e-6", "./models/full-to-full-volumes/70-30/baseline-gmm/disp-rot-1e-6/model")

    predict_mr_us_file("f2f_70-30_fpt-cd_disp-rot-1e-5-best", "./models/full-to-full-volumes/70-30/fpt-cd/disp-rot-1e-5/model-best")
    predict_mr_us_file("f2f_70-30_fpt-cd_disp-rot-1e-6-best", "./models/full-to-full-volumes/70-30/fpt-cd/disp-rot-1e-6/model-best")
    predict_mr_us_file("f2f_70-30_fpt-cd_disp-rot-1e-5", "./models/full-to-full-volumes/70-30/fpt-cd/disp-rot-1e-5/model")
    predict_mr_us_file("f2f_70-30_fpt-cd_disp-rot-1e-6", "./models/full-to-full-volumes/70-30/fpt-cd/disp-rot-1e-6/model")

    predict_mr_us_file("f2f_70-30_fpt-gmm_disp-rot-1e-5-best", "./models/full-to-full-volumes/70-30/fpt-gmm/disp-rot-1e-5/model-best")
    predict_mr_us_file("f2f_70-30_fpt-gmm_disp-rot-1e-6-best", "./models/full-to-full-volumes/70-30/fpt-gmm/disp-rot-1e-6/model-best")
    predict_mr_us_file("f2f_70-30_fpt-gmm_disp-rot-1e-5", "./models/full-to-full-volumes/70-30/fpt-gmm/disp-rot-1e-5/model")
    predict_mr_us_file("f2f_70-30_fpt-gmm_disp-rot-1e-6", "./models/full-to-full-volumes/70-30/fpt-gmm/disp-rot-1e-6/model")

    # 60-30
    predict_mr_us_file("f2f_60-30_baseline-cd_disp-rot", "./models/full-to-full/60-30/baseline-cd/disp-rot/model-best")

    predict_mr_us_file("f2f_60-30_baseline-gmm_disp-rot", "./models/full-to-full/60-30/baseline-gmm/disp-rot/model-best")

    predict_mr_us_file("f2f_60-30_fpt-cd_disp-rot", "./models/full-to-full/60-30/fpt-cd/disp-rot/model-best")

    predict_mr_us_file("f2f_60-30_fpt-gmm_disp-rot", "./models/full-to-full/60-30/fpt-gmm/disp-rot/model-best")

    # 50-30
    predict_mr_us_file("f2f_50-30_baseline-cd_disp-rot", "./models/full-to-full/50-30/baseline-cd/disp-rot/model-best")

    predict_mr_us_file("f2f_50-30_baseline-gmm_disp-rot", "./models/full-to-full/50-30/baseline-gmm/disp-rot/model-best")

    predict_mr_us_file("f2f_50-30_fpt-cd_disp-rot", "./models/full-to-full/50-30/fpt-cd/disp-rot/model-best")

    predict_mr_us_file("f2f_50-30_fpt-gmm_disp-rot", "./models/full-to-full/50-30/fpt-gmm/disp-rot/model-best")

    # 40-30
    predict_mr_us_file("f2f_40-30_baseline-cd_disp-rot", "./models/full-to-full/40-30/baseline-cd/disp-rot/model-best")

    predict_mr_us_file("f2f_40-30_baseline-gmm_disp-rot", "./models/full-to-full/40-30/baseline-gmm/disp-rot/model-best")

    predict_mr_us_file("f2f_40-30_fpt-cd_disp-rot", "./models/full-to-full/40-30/fpt-cd/disp-rot/model-best")

    predict_mr_us_file("f2f_40-30_fpt-gmm_disp-rot", "./models/full-to-full/40-30/fpt-gmm/disp-rot/model-best")

    # 30-30
    predict_mr_us_file("f2f_30-30_baseline-cd_disp-rot", "./models/full-to-full/30-30/baseline-cd/disp-rot/model-best")

    predict_mr_us_file("f2f_30-30_baseline-gmm_disp-rot", "./models/full-to-full/30-30/baseline-gmm/disp-rot/model-best")

    predict_mr_us_file("f2f_30-30_fpt-cd_disp-rot", "./models/full-to-full/30-30/fpt-cd/disp-rot/model-best")

    predict_mr_us_file("f2f_30-30_fpt-gmm_disp-rot", "./models/full-to-full/30-30/fpt-gmm/disp-rot/model-best")
    '''

    '''
    # Sweeps

    for sweep in [3, 4, 5, 6]:
        print()
        print()
        print(str(sweep))
        print()
        print()
        # 70-30
        predict_mr_us_file_Sweep("p2f_70-30_s" + str(sweep) + "_baseline-cd_disp-rot", "./models/sparse-to-full/70-30/sweep" + str(sweep) + "/baseline-cd/disp-rot/model-best", sweep)

        predict_mr_us_file_Sweep("p2f_70-30_s" + str(sweep) + "_baseline-gmm_disp-rot", "./models/sparse-to-full/70-30/sweep" + str(sweep) + "/baseline-gmm/disp-rot/model-best", sweep)

        predict_mr_us_file_Sweep("p2f_70-30_s" + str(sweep) + "_fpt-cd_disp-rot", "./models/sparse-to-full/70-30/sweep" + str(sweep) + "/fpt-cd/disp-rot/model-best", sweep)

        predict_mr_us_file_Sweep("p2f_70-30_s" + str(sweep) + "_fpt-gmm_disp-rot", "./models/sparse-to-full/70-30/sweep" + str(sweep) + "/fpt-gmm/disp-rot/model-best", sweep)

        # 60-30
        predict_mr_us_file_Sweep("p2f_60-30_s" + str(sweep) + "_baseline-cd_disp-rot", "./models/sparse-to-full/60-30/sweep" + str(sweep) + "/baseline-cd/disp-rot/model-best", sweep)

        predict_mr_us_file_Sweep("p2f_60-30_s" + str(sweep) + "_baseline-gmm_disp-rot", "./models/sparse-to-full/60-30/sweep" + str(sweep) + "/baseline-gmm/disp-rot/model-best", sweep)

        predict_mr_us_file_Sweep("p2f_60-30_s" + str(sweep) + "_fpt-cd_disp-rot", "./models/sparse-to-full/60-30/sweep" + str(sweep) + "/fpt-cd/disp-rot/model-best", sweep)

        predict_mr_us_file_Sweep("p2f_60-30_s" + str(sweep) + "_fpt-gmm_disp-rot", "./models/sparse-to-full/60-30/sweep" + str(sweep) + "/fpt-gmm/disp-rot/model-best", sweep)

        # 50-30
        predict_mr_us_file_Sweep("p2f_50-30_s" + str(sweep) + "_baseline-cd_disp-rot", "./models/sparse-to-full/50-30/sweep" + str(sweep) + "/baseline-cd/disp-rot/model-best", sweep)

        predict_mr_us_file_Sweep("p2f_50-30_s" + str(sweep) + "_baseline-gmm_disp-rot", "./models/sparse-to-full/50-30/sweep" + str(sweep) + "/baseline-gmm/disp-rot/model-best", sweep)

        predict_mr_us_file_Sweep("p2f_50-30_s" + str(sweep) + "_fpt-cd_disp-rot", "./models/sparse-to-full/50-30/sweep" + str(sweep) + "/fpt-cd/disp-rot/model-best", sweep)

        predict_mr_us_file_Sweep("p2f_50-30_s" + str(sweep) + "_fpt-gmm_disp-rot", "./models/sparse-to-full/50-30/sweep" + str(sweep) + "/fpt-gmm/disp-rot/model-best", sweep)

        # 40-30
        predict_mr_us_file_Sweep("p2f_40-30_s" + str(sweep) + "_baseline-cd_disp-rot", "./models/sparse-to-full/40-30/sweep" + str(sweep) + "/baseline-cd/disp-rot/model-best", sweep)

        predict_mr_us_file_Sweep("p2f_40-30_s" + str(sweep) + "_baseline-gmm_disp-rot", "./models/sparse-to-full/40-30/sweep" + str(sweep) + "/baseline-gmm/disp-rot/model-best", sweep)

        predict_mr_us_file_Sweep("p2f_40-30_s" + str(sweep) + "_fpt-cd_disp-rot", "./models/sparse-to-full/40-30/sweep" + str(sweep) + "/fpt-cd/disp-rot/model-best", sweep)

        predict_mr_us_file_Sweep("p2f_40-30_s" + str(sweep) + "_fpt-gmm_disp-rot", "./models/sparse-to-full/40-30/sweep" + str(sweep) + "/fpt-gmm/disp-rot/model-best", sweep)

        # 30-30
        predict_mr_us_file_Sweep("p2f_30-30_s" + str(sweep) + "_baseline-cd_disp-rot", "./models/sparse-to-full/30-30/sweep" + str(sweep) + "/baseline-cd/disp-rot/model-best", sweep)

        predict_mr_us_file_Sweep("p2f_30-30_s" + str(sweep) + "_baseline-gmm_disp-rot", "./models/sparse-to-full/30-30/sweep" + str(sweep) + "/baseline-gmm/disp-rot/model-best", sweep)

        predict_mr_us_file_Sweep("p2f_30-30_s" + str(sweep) + "_fpt-cd_disp-rot", "./models/sparse-to-full/30-30/sweep" + str(sweep) + "/fpt-cd/disp-rot/model-best", sweep)

        predict_mr_us_file_Sweep("p2f_30-30_s" + str(sweep) + "_fpt-gmm_disp-rot", "./models/sparse-to-full/30-30/sweep" + str(sweep) + "/fpt-gmm/disp-rot/model-best", sweep)
    '''