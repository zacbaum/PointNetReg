from data_loader import DataGenerator
from model import PointRegNet
from keras.optimizers import SGD
import os
import matplotlib
matplotlib.use('AGG')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


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

def main():
    train_file = './ModelNet40/ply_data_train.h5'
    test_file = './ModelNet40/ply_data_test.h5'

    epochs = 100
    batch_size = 16 * 3

    train = DataGenerator(train_file, batch_size, train=True)
    val = DataGenerator(test_file, batch_size, train=False)

    '''
    data = []     # store all the generated data batches
    labels = []   # store all the generated label batches
    max_iter = 10  # maximum number of iterations, in each iteration one batch is generated; the proper value depends on batch size and size of whole data
    i = 0
    for d, l in train.generator():
        data.append(d)
        labels.append(l)
        i += 1
        if i == max_iter:
            break

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x1 = [i[0] for i in data[0][0][0]]
    y1 = [i[1] for i in data[0][0][0]]
    z1 = [i[2] for i in data[0][0][0]]

    x2 = [i[0] for i in data[0][1][0]]
    y2 = [i[1] for i in data[0][1][0]]
    z2 = [i[2] for i in data[0][1][0]]

    ax.scatter(x1, y1, z1, c='r', marker='o')
    ax.scatter(x2, y2, z2, c='b', marker='o')

    plt.show()
    plt.savefig('pre_reg-scatter.png', dpi=1000)
    plt.close()
    '''

    model = PointRegNet(2048, 2048)
    lr = 0.0001
    opt = SGD(lr=lr)
    model.compile(optimizer=opt,
                  loss='mean_squared_error',
                  metrics=['mse'])
    if not os.path.exists('./results/'):
        os.mkdir('./results/')

    history = model.fit_generator(train.generator(),
                                  steps_per_epoch=9840 // batch_size,
                                  epochs=epochs,
                                  validation_data=val.generator(),
                                  validation_steps=2468 // batch_size,
                                  verbose=1)

    plot_history(history, './results/')
    model.save_weights('./results/pointnet_weights.h5')

if __name__ == '__main__':
    main()
