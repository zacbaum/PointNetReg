from data_loader import DataGenerator
from keras.optimizers import SGD
from losses import sorted_mse_loss
from model import ConditionalTransformerNet
from mpl_toolkits.mplot3d import Axes3D
import h5py
import keras
import matplotlib
import matplotlib.pyplot as plt
import os
import tensorflow as tf
matplotlib.use('AGG')

def plot_history(history, result_dir):
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(os.path.join(result_dir, 'model_accuracy.png'))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss.png'))
    plt.close()


class Prediction_Plotter(keras.callbacks.Callback):
    def __init__(self, X_val, Y_val):
        self.X_val = X_val
        self.Y_val = Y_val

    def on_train_begin(self, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):    
        plt.clf()
        pred = self.model.predict(self.X_val)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x_true = [i[0] for i in self.Y_val]
        y_true = [i[1] for i in self.Y_val]
        z_true = [i[2] for i in self.Y_val]

        x_pred = [i[0] for i in pred]
        y_pred = [i[1] for i in pred]
        z_pred = [i[2] for i in pred]

        ax.scatter(x_true, y_true, z_true, c='y', marker='.')
        ax.scatter(x_pred, y_pred, z_pred, c='g', marker='.')

        plt.show()
        plt.savefig('in_training_reg-scatter.png', dpi=1000)
        plt.close()


def main():
    train_file = './ModelNet40/ply_data_train.h5'
    test_file = './ModelNet40/ply_data_test.h5'

    epochs = 10
    batch_size = 32 * 3

    train = DataGenerator(train_file, batch_size, train=True)
    val = DataGenerator(test_file, batch_size, train=False)

    data = []     # store all the generated data batches
    labels = []   # store all the generated ground_truth batches
    max_iter = 1  # maximum number of iterations, in each iteration one batch is generated; the proper value depends on batch size and size of whole data
    i = 0
    for d, l in train.generator():
        data.append(d)
        labels.append(l)
        i += 1
        if i == max_iter:
            break

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x1 = [i[0] for i in data[0][0][0]] # Fixed X (Red)
    y1 = [i[1] for i in data[0][0][0]] # Fixed Y (Red)
    z1 = [i[2] for i in data[0][0][0]] # Fixed Z (Red)

    x2 = [i[0] for i in data[0][1][0]] # Moved X (Blue)
    y2 = [i[1] for i in data[0][1][0]] # Moved Y (Blue)
    z2 = [i[2] for i in data[0][1][0]] # Moved Z (Blue)

    x3 = [i[0] for i in data[0][2][0]] # Moved X (Green)
    y3 = [i[1] for i in data[0][2][0]] # Moved Y (Green)
    z3 = [i[2] for i in data[0][2][0]] # Moved Z (Green)

    x4 = [i[0] for i in labels[0][0]]  # Ground Truth X (Yellow)
    y4 = [i[1] for i in labels[0][0]]  # Ground Truth Y (Yellow)
    z4 = [i[2] for i in labels[0][0]]  # Ground Truth Z (Yellow)

    ax.scatter(x1, y1, z1, c='r', marker='.')
    ax.scatter(x2, y2, z2, c='b', marker='.')
    ax.scatter(x3, y3, z3, c='g', marker='.')
    ax.scatter(x4, y4, z4, c='y', marker='.')

    plt.show()
    plt.savefig('pre_reg-scatter.png', dpi=1000)
    plt.close()


    fixed_len = data[0][0].shape[1]
    moving_len = data[0][1].shape[1]
    conditional_len = data[0][2].shape[1]
    f = h5py.File(train_file, mode='r')
    num_train = f['data'].shape[0]
    f = h5py.File(test_file, mode='r')
    num_val = f['data'].shape[0]

    model = ConditionalTransformerNet(fixed_len, moving_len, conditional_len)
    lr = 0.01
    opt = SGD(lr=lr)
    model.compile(optimizer=opt,
                  loss=sorted_mse_loss)
    if not os.path.exists('./results/'):
        os.mkdir('./results/')

    first_val_X = data[0]
    first_val_Y = labels[0][0]
    Prediction_Plot = Prediction_Plotter(first_val_X, first_val_Y)
    history = model.fit_generator(train.generator(),
                                  steps_per_epoch=num_train // batch_size,
                                  epochs=epochs,
                                  validation_data=val.generator(),
                                  validation_steps=num_val // batch_size,
                                  callbacks=[Prediction_Plot],
                                  verbose=1)

    plot_history(history, './results/')
    model.save_weights('./results/pointnet_weights.h5')

if __name__ == '__main__':
    main()
