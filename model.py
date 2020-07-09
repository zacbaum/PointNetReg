import tensorflow as tf
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    Input,
    Dense,
    RepeatVector,
    Reshape,
    concatenate,
    add,
)
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
import numpy as np


class MatMul(Layer):
    def __init__(self, **kwargs):
        super(MatMul, self).__init__(**kwargs)

    def build(self, input_shape):
        # Used purely for shape validation.
        if not isinstance(input_shape, list):
            raise ValueError("`MatMul` layer should be called " "on a list of inputs")
        if len(input_shape) != 2:
            raise ValueError(
                "The input of `MatMul` layer should be a list containing 2 elements"
            )

        if len(input_shape[0]) != 3 or len(input_shape[1]) != 3:
            raise ValueError("The dimensions of each element of inputs should be 3")

        if input_shape[0][-1] != input_shape[1][1]:
            raise ValueError(
                "The last dimension of inputs[0] should match the dimension 1 of inputs[1]"
            )

    def call(self, inputs):
        if not isinstance(inputs, list):
            raise ValueError(
                "A `MatMul` layer should be called " "on a list of inputs."
            )
        import tensorflow as tf

        return tf.matmul(inputs[0], inputs[1])

    def compute_output_shape(self, input_shape):
        output_shape = [input_shape[0][0], input_shape[0][1], input_shape[1][-1]]
        return tuple(output_shape)


def PointNet_features(input_len, dimensions=3, filters=[64, 128, 1024]):

    input_points = Input(shape=(input_len, dimensions))
    # input transformation net
    x = Conv1D(filters[0], 1, activation="relu")(input_points)
    for i in filters[1:]:
        x = Conv1D(i, 1, activation="relu")(x)
    x = MaxPooling1D(pool_size=input_len)(x)

    for i in [512, 256]:
        x = Dense(i, activation="relu")(x)

    x = Dense(
        dimensions * dimensions,
        weights=[
            np.zeros([256, dimensions * dimensions]),
            np.eye(dimensions).flatten().astype(np.float32),
        ],
    )(x)
    input_T = Reshape((dimensions, dimensions))(x)

    # forward net
    g = MatMul()([input_points, input_T])
    for i in [64, 64]:
        g = Conv1D(i, 1, activation="relu")(g)

    # feature transform net
    f = Conv1D(filters[0], 1, activation="relu")(g)
    for i in filters[1:]:
        f = Conv1D(i, 1, activation="relu")(f)
    f = MaxPooling1D(pool_size=input_len)(f)
    for i in [512, 256]:
        f = Dense(i, activation="relu")(f)
    f = Dense(
        64 * 64,
        weights=[np.zeros([256, 64 * 64]), np.eye(64).flatten().astype(np.float32)],
    )(f)
    feature_T = Reshape((64, 64))(f)

    # forward net
    g = MatMul()([g, feature_T])
    g = Conv1D(filters[0], 1, activation="relu")(g)
    for i in filters[1:]:
        g = Conv1D(i, 1, activation="relu")(g)

    # global feature
    global_feature = MaxPooling1D(pool_size=input_len)(g)
    global_feature = Reshape((filters[-1],))(global_feature)

    model = Model(inputs=input_points, outputs=global_feature, name="PointNet")

    return model


def Point_features(input_len, dimensions=3, filters=[64, 128, 1024]):

    input_points = Input(shape=(input_len, dimensions))
    # input transformation net
    x = Conv1D(filters[0], 1, activation="relu")(input_points)
    for i in filters[1:]:
        x = Conv1D(i, 1, activation="relu")(x)
    x = MaxPooling1D(pool_size=input_len)(x)
    x = Reshape((filters[-1],))(x)

    model = Model(inputs=input_points, outputs=x, name="1DCNN")

    return model


def ConvTransformerBlock(input_tensor, ct_activation, filters, skips):

    x = input_tensor

    x = Conv1D(filters, 1, activation=ct_activation)(x)

    if skips:  # Add 4 more layers, shortcut and add layer.
        x = Conv1D(filters, 1, activation=ct_activation)(x)
        x = Conv1D(filters, 1, activation=ct_activation)(x)
        x = Conv1D(filters, 1, activation=ct_activation)(x)
        x = Conv1D(filters, 1, activation=ct_activation)(x)

        x_short = Conv1D(filters, 1, activation=ct_activation)(input_tensor)
        x = add([x, x_short])

    return x


def FreePointTransformer(
    num_points,
    dims=3,
    ct_activation="relu",
    pn_filters=[64, 128, 1024],
    ctn_filters=[1024, 512, 256, 128, 64],
    skips=False,
):

    fixed = Input(shape=(num_points, dims), name="Fixed_Model")
    moved = Input(shape=(num_points, dims), name="Moved_Model")
    moving = Input(shape=(num_points, dims), name="Moving_Model")

    pointNet = PointNet_features(int(num_points), dims, pn_filters)

    fixed_pointNet = pointNet(fixed)
    moving_pointNet = pointNet(moved)

    point_features = concatenate([fixed_pointNet, moving_pointNet])
    point_features_matrix = RepeatVector(num_points)(point_features)

    out = moving
    x = concatenate([point_features_matrix, out])

    for num_filters in ctn_filters:
        x = ConvTransformerBlock(x, ct_activation, num_filters, skips)

    x = Conv1D(dims, 1)(x)
    out = add([x, out])

    model = Model(inputs=[fixed, moved, moving], outputs=out, name="FPT")

    plot_model(model, to_file="model.png", show_shapes=True)  # , expand_nested=True)

    return model
