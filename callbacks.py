from mpl_toolkits.mplot3d import Axes3D
import keras
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('AGG')
import numpy as np

class Prediction_Plotter(keras.callbacks.Callback):
    def __init__(self, X_val, Y_val, fname_tag, debug=False):
        self.X_val = X_val
        self.Y_val = Y_val
        self.fname_tag = fname_tag
        self.debug = debug

    def on_train_begin(self, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):    
        pred = self.model.predict(self.X_val)
        
        assert(self.Y_val.shape[0] == pred.shape[0])
        for batch_id in range(0, pred.shape[0], 10):
            
            plt.clf()
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            x_true = [i[0] for i in self.Y_val[batch_id]]
            y_true = [i[1] for i in self.Y_val[batch_id]]
            z_true = [i[2] for i in self.Y_val[batch_id]]

            x_pred = [i[0] for i in pred[batch_id]]
            y_pred = [i[1] for i in pred[batch_id]]
            z_pred = [i[2] for i in pred[batch_id]]
            
            ax.scatter(x_true, y_true, z_true, c='y', marker='.')
            ax.scatter(x_pred, y_pred, z_pred, c='g', marker='.')

            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])

            plt.show()
            plt.savefig('./results/' + str(self.fname_tag) + '_id_' + str(batch_id + 1) + '_epoch_' + str(epoch + 1) + '_reg-scatter.png', dpi=250)
            plt.close()

            if self.debug:
                plt.clf()
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

                ax.scatter(x_pred, y_pred, z_pred, c='g', marker='.')

                mn = abs(np.min(pred[batch_id]))
                mx = np.max(pred[batch_id])
                limit = max(mn, mx)

                ax.set_xlim([limit, -limit])
                ax.set_ylim([limit, -limit])
                ax.set_zlim([limit, -limit])

                plt.show()
                plt.savefig('./results/PRED_ZOOMED-' + str(self.fname_tag) + '_id_' + str(batch_id + 1) + '_epoch_' + str(epoch + 1) + '_reg-scatter.png', dpi=250)
                plt.close()