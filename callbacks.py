from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import tensorflow.keras
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('AGG')
import numpy as np

class Prediction_Plotter(tf.keras.callbacks.Callback):
    def __init__(self, X, Y, fname_tag, debug=False):
        self.X = X
        self.Y = Y
        self.fname_tag = fname_tag
        self.debug = debug

    def on_train_begin(self, logs={}):
        self.on_epoch_end(-1)

    def on_epoch_end(self, epoch, logs={}):

        pred = self.model.predict_on_batch(self.X)

        assert(self.Y.shape[0] == pred.shape[0])
        for batch_id in range(0, pred.shape[0], int(pred.shape[0] / 5)):
            
            plt.clf()
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            x_true = [i[0] for i in self.Y[batch_id]]
            y_true = [i[1] for i in self.Y[batch_id]]
            z_true = [i[2] for i in self.Y[batch_id]]
            '''
            x_f = [i[0] for i in self.X[0][batch_id]]
            y_f = [i[1] for i in self.X[0][batch_id]]
            z_f = [i[2] for i in self.X[0][batch_id]]

            x_m = [i[0] for i in self.X[1][batch_id]]
            y_m = [i[1] for i in self.X[1][batch_id]]
            z_m = [i[2] for i in self.X[1][batch_id]]

            x_r = [i[0] for i in self.X[2][batch_id]]
            y_r = [i[1] for i in self.X[2][batch_id]]
            z_r = [i[2] for i in self.X[2][batch_id]]
            '''
            x_pred = [i[0] for i in pred[batch_id]]
            y_pred = [i[1] for i in pred[batch_id]]
            z_pred = [i[2] for i in pred[batch_id]]
            
            ax.scatter(x_true, y_true, z_true, c='y', marker='.')
            ax.scatter(x_pred, y_pred, z_pred, c='g', marker='.')

            #ax.scatter(x_f, y_f, z_f, c='r', marker='*', alpha=0.1)
            #ax.scatter(x_m, y_m, z_m, c='b', marker='*', alpha=0.1)
            #ax.scatter(x_r, y_r, z_r, c='k', marker='*', alpha=0.1)

            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])

            plt.show()
            plt.savefig(str(self.fname_tag) + '_id_' + str(batch_id + 1) + '_epoch_' + str(epoch + 1) + '_reg-scatter.png', dpi=100)
            plt.close()