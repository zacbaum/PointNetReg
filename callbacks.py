from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import tensorflow.keras
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('AGG')
import numpy as np

class Prediction_Plotter(tf.keras.callbacks.Callback):
    def __init__(self, X, Y, fname_tag, scale=1, debug=False):
        self.X = X
        self.Y = Y
        self.fname_tag = fname_tag
        self.debug = debug
        self.scale = scale

    def on_train_begin(self, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        if epoch % 1 == 0:
            pred = self.model.predict(self.X)

            assert(self.Y.shape[0] == pred.shape[0])
            for batch_id in range(0, pred.shape[0], int(pred.shape[0] / 5)):
                
                plt.clf()
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

                x_true = [i[0] for i in self.X[0][batch_id]]
                y_true = [i[1] for i in self.X[0][batch_id]]
                z_true = [i[2] for i in self.X[0][batch_id]]

                x_pred = [i[0] for i in pred[batch_id]]
                y_pred = [i[1] for i in pred[batch_id]]
                z_pred = [i[2] for i in pred[batch_id]]
                
                ax.scatter(x_true, y_true, z_true, c='y', marker='.')
                ax.scatter(x_pred, y_pred, z_pred, c='g', marker='.')

                ax.set_xlim([-1 * self.scale, 1 * self.scale])
                ax.set_ylim([-1 * self.scale, 1 * self.scale])
                ax.set_zlim([-1 * self.scale, 1 * self.scale])

                plt.show()
                plt.savefig(str(self.fname_tag) + '_id_' + str(batch_id + 1) + '_epoch_' + str(epoch + 1) + '_reg-scatter.png', dpi=150)
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
                    plt.savefig('./results/PRED_ZOOMED-' + str(self.fname_tag) + '_id_' + str(batch_id + 1) + '_epoch_' + str(epoch + 1) + '_reg-scatter.png', dpi=150)
                    plt.close()


                    plt.clf()
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')

                    x_move = [i[0] for i in self.X[1][batch_id]]
                    y_move = [i[1] for i in self.X[1][batch_id]]
                    z_move = [i[2] for i in self.X[1][batch_id]]

                    ax.scatter(x_move, y_move, z_move, c='r', marker='.')
                    ax.scatter(x_pred, y_pred, z_pred, c='g', marker='.')
                
                    ax.set_xlim([-1, 1])
                    ax.set_ylim([-1, 1])
                    ax.set_zlim([-1, 1])

                    plt.show()
                    plt.savefig('./results/MOVE_PRED-' + str(self.fname_tag) + '_id_' + str(batch_id + 1) + '_epoch_' + str(epoch + 1) + '_reg-scatter.png', dpi=150)
                    plt.close()

class PlotLosses(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):

        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        f, ax1 = plt.subplots(1, 1, sharex=True)
                
        ax1.plot(self.x, self.losses, label="loss")
        ax1.plot(self.x, self.val_losses, label="validation loss")
        ax1.legend()
        
        plt.show()
        plt.savefig('live_metrics.png', dpi=250)
        plt.yscale('log')
        plt.savefig('live_metrics-log.png', dpi=250)
        plt.close()