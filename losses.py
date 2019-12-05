import numpy as np
from keras import backend as K
import tensorflow as tf

def sort(a, col):
    return tf.gather(a, tf.nn.top_k(-a[:, col], k=a.get_shape()[0].value).indices)

def sorted_mse_loss(y_true, y_pred):

    if not K.is_tensor(y_pred):
        y_pred = K.constant(y_pred)

    #y_true_sorted = tf.sort(y_true, axis=-1, direction='ASCENDING')
    #y_pred_sorted = tf.sort(y_pred, axis=-1, direction='ASCENDING')

    y_true_sorted = sort(sort(sort(y_true, 2), 1), 0)
    y_pred_sorted = sort(sort(sort(y_pred, 2), 1), 0)

    y_true_sorted = K.cast(y_true_sorted, y_pred_sorted.dtype)
    
    return K.mean(K.square(y_pred_sorted - y_true_sorted), axis=-1)