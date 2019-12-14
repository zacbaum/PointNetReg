from callbacks import Prediction_Plotter
from data_loader import DataGenerator
from keras.optimizers import SGD, Adam
from losses import sorted_mse_loss, nll, sorted_nll, kl_divergence, sorted_kl_divergence
from model import ConditionalTransformerNet
from mpl_toolkits.mplot3d import Axes3D
import h5py
import keras
import matplotlib
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import tensorflow_probability as tfp

from keras import backend as K
import numpy as np

matplotlib.use('AGG')

def plot_results(attribute, output, filename):
    plt.figure(figsize=(16,10))
    for name, history in output:
        val = plt.plot(history.epoch, history.history['val_' + attribute], '--', label=name.title()+' Val')
        plt.plot(history.epoch, history.history[attribute], color=val[0].get_color(), label=name.title()+' Train')
    plt.xlabel('Epochs')
    plt.ylabel(attribute)
    plt.legend()
    plt.xlim([0, max(history.epoch)])
    plt.savefig(filename + '.png', dpi=250)
    plt.close()

def main():
    train_file = './ModelNet40/ply_data_train.h5'
    test_file = './ModelNet40/ply_data_test.h5'

    num_epochs = 10
    batch_size = 32 * 3

    train = DataGenerator(train_file, batch_size, train=True)
    
    train_data = []     # store all the generated data batches
    train_labels = []   # store all the generated ground_truth batches
    max_iter = 1        # maximum number of iterations, in each iteration one batch is generated; the proper value depends on batch size and size of whole data
    i = 0
    for d, l in train.generator():
        train_data.append(d)
        train_labels.append(l)
        i += 1
        if i == max_iter:
            break
    fixed_len = train_data[0][0].shape[1]
    moving_len = train_data[0][1].shape[1]

    assert fixed_len == moving_len, 'Lengths not consistent'
    num_points = fixed_len

    first_train_X = train_data[0]
    first_train_Y = train_labels[0]
    Prediction_Plot_Train = Prediction_Plotter(first_train_X, first_train_Y, 'train')

    val = DataGenerator(test_file, batch_size, train=False)

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
    Prediction_Plot_Val = Prediction_Plotter(first_val_X, first_val_Y, 'val')

    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x1 = [i[0] for i in val_data[0][0][0]] # Fixed X (Red)
    y1 = [i[1] for i in val_data[0][0][0]] # Fixed Y (Red)
    z1 = [i[2] for i in val_data[0][0][0]] # Fixed Z (Red)

    x2 = [i[0] for i in val_data[0][1][0]] # Moved X (Blue)
    y2 = [i[1] for i in val_data[0][1][0]] # Moved Y (Blue)
    z2 = [i[2] for i in val_data[0][1][0]] # Moved Z (Blue)

    x4 = [i[0] for i in val_labels[0][0]]  # Ground Truth X (Yellow)
    y4 = [i[1] for i in val_labels[0][0]]  # Ground Truth Y (Yellow)
    z4 = [i[2] for i in val_labels[0][0]]  # Ground Truth Z (Yellow)

    ax.scatter(x1, y1, z1, c='r', marker='.')
    ax.scatter(x2, y2, z2, c='b', marker='.')
    ax.scatter(x4, y4, z4, c='y', marker='.')

    plt.show()
    plt.savefig('pre_reg-scatter.png', dpi=250)
    plt.close()
    '''

    model = ConditionalTransformerNet(num_points, ct_activation='linear', dropout=0., verbose=False)
    max_learning_rate = 0.05
    #min_learning_rate = 5e-5
    #learning_rate_decay = (max_learning_rate - min_learning_rate) / num_epochs
    opt = SGD(lr=max_learning_rate)#, decay=learning_rate_decay)
    model.compile(optimizer=opt,
                  loss=sorted_mse_loss)
    if not os.path.exists('./results/'):
        os.mkdir('./results/')

    f = h5py.File(train_file, mode='r')
    num_train = f['data'].shape[0]
    f = h5py.File(test_file, mode='r')
    num_val = f['data'].shape[0]

    history = model.fit_generator(train.generator(),
                                  steps_per_epoch=num_train // batch_size,
                                  epochs=num_epochs,
                                  validation_data=val.generator(),
                                  validation_steps=num_val // batch_size,
                                  callbacks=[Prediction_Plot_Train, Prediction_Plot_Val],
                                  verbose=1)
    model.save('./results/CTN.h5')
    name = ''
    output = [(name, history)]
    plot_results('loss', output, 'loss') 

if __name__ == '__main__':
    main()

'''
    first_val_X_moved = tf.convert_to_tensor(first_val_X[1][0], np.float32)
    first_val_Y = tf.convert_to_tensor(first_val_Y[0], np.float32)
    
    first_val_X_moved = np.array([[0, 0, 2],
                                  [0, 1, 2],
                                  [1, 2, 0],
                                  [2, 0, 1]])
    first_val_Y = np.array([[1, 0, 2],
                            [0, 0, 1],
                            [0, 0, 1],
                            [0, 2, 1]])

    first_val_X_moved = tf.convert_to_tensor(first_val_X_moved, np.float32)
    first_val_Y = tf.convert_to_tensor(first_val_Y, np.float32)

    sums = tf.add(tf.slice(first_val_X_moved, [0, 0], [-1, 1]) * 100, tf.add(tf.slice(first_val_X_moved, [0, 1], [-1, 1]) * 10, tf.slice(first_val_X_moved, [0, 2], [-1, 1])))
    reordered = tf.gather(first_val_X_moved, tf.nn.top_k(sums[:, 0], k=tf.shape(first_val_X_moved)[0], sorted=False).indices)
    first_val_X_moved = tf.reverse(reordered, axis=[0])

    sums = tf.add(tf.slice(first_val_Y, [0, 0], [-1, 1]) * 100, tf.add(tf.slice(first_val_Y, [0, 1], [-1, 1]) * 10, tf.slice(first_val_Y, [0, 2], [-1, 1])))
    reordered = tf.gather(first_val_Y, tf.nn.top_k(sums[:, 0], k=tf.shape(first_val_Y)[0], sorted=False).indices)
    first_val_Y = tf.reverse(reordered, axis=[0])

    tfd = tfp.distributions
    std = K.std(first_val_Y)
    likelihood1 = tfd.Normal(loc=first_val_Y, scale=std)
    std = K.std(first_val_X_moved)
    likelihood2 = tfd.Normal(loc=first_val_X_moved, scale=std)

    print(K.eval(first_val_X_moved))
    print()
    print(K.eval(first_val_Y))
    print()
    print(K.eval(likelihood1.kl_divergence(likelihood2)))
    print(K.eval(K.mean(likelihood1.kl_divergence(likelihood2))))
    print(K.eval(K.mean(K.mean(likelihood1.kl_divergence(likelihood2), axis=-1))))
'''