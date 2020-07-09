import h5py
import tensorflow as tf
import tensorflow.keras
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
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from losses import chamfer_loss, chamfer_loss_batch, gmm_nll_loss
from model import FreePointTransformer, MatMul
from mpl_toolkits.mplot3d import Axes3D
from wandb.keras import WandbCallback

matplotlib.use("AGG")
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]

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


def train(load, batch_size, learning_rate, rotate, displace, deform, part_nn, epochs):
    train_file = "./ModelNet40/ply_data_train.h5"
    train = h5py.File(train_file, mode="r")
    test_file = "./ModelNet40/ply_data_test.h5"
    test = h5py.File(test_file, mode="r")

    if not os.path.exists("./results" + str(sys.argv[2]) + "/"):
        os.mkdir("./results" + str(sys.argv[2]) + "/")

    loss_name = str(sys.argv[1])
    loss_func = None
    metrics = []

    if loss_name == "chamfer_loss":
        loss_func = chamfer_loss

    if loss_name == "chamfer_loss_batch":
        loss_func = chamfer_loss_batch

    if loss_name == "gmm_nll_loss":
        loss_func = gmm_nll_loss(0.01, 0.1)
        metrics = [chamfer_loss]

    train = DataGenerator(
        train,
        batch_size,
        shuffle=True,
        rotate=rotate,
        displace=displace,
        deform=deform,
        dims=4,
        part=1,
        part_nn=int(2048 * part_nn),
        shuffle_points=True,
    )

    val = DataGenerator(
        test,
        batch_size,
        shuffle=False,
        rotate=rotate,
        displace=displace,
        deform=deform,
        dims=4,
        part=1,
        part_nn=int(2048 * part_nn),
        shuffle_points=True,
    )

    val_data = []  # store all the generated data batches
    val_labels = []  # store all the generated ground_truth batches
    max_iter = 1  # maximum number of iterations, in each iteration one batch is generated; the proper value depends on batch size and size of whole data
    i = 0
    for d, l in val:
        val_data.append(d)
        val_labels.append(l)
        i += 1
        if i == max_iter:
            break

    first_val_X = val_data[0]
    first_val_Y = val_labels[0]
    Prediction_Plot_Val = Prediction_Plotter(
        first_val_X,
        first_val_Y,
        "./results" + str(sys.argv[2]) + "/" + loss_name + "-val",
    )
    fixed_len = val_data[0][0].shape[1]
    moving_len = val_data[0][1].shape[1]
    assert fixed_len == moving_len
    num_points = fixed_len

    if not load:
        wandb.init(project="ctn-chamfer", name="PN 4D Curr", id="0221")
    else:
        wandb.init(project="ctn-chamfer", name="PN 4D Curr", resume="0221")

    model = FreePointTransformer(
        num_points,
        dims=4,
        pn_filters=[64, 128, 1024],
        ctn_filters=[1024, 512, 256, 128, 64],
        skips=False,
    )
    initial_epoch = 0
    if load:
        model = load_model(
            wandb.restore("model-best.h5").name,
            custom_objects={"MatMul": MatMul, loss_name: loss_func},
        )
        initial_epoch = wandb.run.step

    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss=loss_func, metrics=metrics)

    f = h5py.File(train_file, mode="r")
    num_train = f["data"].shape[0]
    f = h5py.File(test_file, mode="r")
    num_val = f["data"].shape[0]

    print("Batch Size:      " + str(batch_size))
    print("LR:              " + str(learning_rate))
    print("Rotation:        " + str(rotate))
    print("Displacement:    " + str(displace))
    print("Deformation:     " + str(deform))
    print("Occlusion:       " + str(100 * (1 - part_nn)) + "%")
    print("Epochs to Train: " + str(epochs) + "\n")

    history = model.fit_generator(
        train,
        steps_per_epoch=num_train // batch_size,
        epochs=initial_epoch + epochs,
        initial_epoch=initial_epoch,
        validation_data=val,
        validation_steps=num_val // batch_size,
        callbacks=[
            Prediction_Plot_Val,
            ModelCheckpoint(os.path.join(wandb.run.dir, "model-checkpoint.h5")),
            WandbCallback(log_weights=True),
        ],
        verbose=1,
    )

    model.save(os.path.join(wandb.run.dir, "model.h5"))


if __name__ == "__main__":

    learning_rate = float(sys.argv[3])
    batch_size = int(sys.argv[4])

    # order, batch_size, learning_rate, rotate, displace, deform, part_nn, epochs
    train(0,  batch_size, learning_rate, 10, 0.00, 0.0, 1.0, 50)
    train(1,  batch_size, learning_rate, 15, 0.00, 0.0, 1.0, 50)
    train(2,  batch_size, learning_rate, 20, 0.00, 0.0, 1.0, 50)
    train(3,  batch_size, learning_rate, 25, 0.00, 0.0, 1.0, 50)
    train(4,  batch_size, learning_rate, 30, 0.00, 0.0, 1.0, 50)
    train(5,  batch_size, learning_rate, 35, 0.00, 0.0, 1.0, 50)
    train(6,  batch_size, learning_rate, 40, 0.00, 0.0, 1.0, 50)
    train(7,  batch_size, learning_rate, 45, 0.00, 0.0, 1.0, 50)
    train(8,  batch_size, learning_rate, 45, 0.25, 0.0, 1.0, 50)
    train(9,  batch_size, learning_rate, 45, 0.50, 0.0, 1.0, 50)
    train(10, batch_size, learning_rate, 45, 0.75, 0.0, 1.0, 50)
    train(11, batch_size, learning_rate, 45, 1.00, 0.0, 1.0, 50)
    train(12, batch_size, learning_rate, 45, 1.00, 0.025, 1.0, 50)
    train(13, batch_size, learning_rate, 45, 1.00, 0.050, 1.0, 50)
    train(14, batch_size, learning_rate, 45, 1.00, 0.075, 1.0, 50)
    train(15, batch_size, learning_rate, 45, 1.00, 0.100, 1.0, 500)

