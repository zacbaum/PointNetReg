import numpy as np
import tensorflow as tf
from keras import backend as K
import tensorflow_probability as tfp

tfd = tfp.distributions


def chamfer_distance(y_true, y_pred):
    if K.int_shape(y_pred)[1] == 4:
        y_pred = y_pred[:, :-1]
        y_true = y_true[:, :-1]

    row_norms_true = tf.reduce_sum(tf.square(y_true), axis=1)
    row_norms_true = tf.reshape(row_norms_true, [-1, 1])
    row_norms_pred = tf.reduce_sum(tf.square(y_pred), axis=1)
    row_norms_pred = tf.reshape(row_norms_pred, [1, -1])
    D = row_norms_true - 2 * tf.matmul(y_true, y_pred, False, True) + row_norms_pred
    dist_t_to_p = K.mean(K.min(D, axis=0))
    dist_p_to_t = K.mean(K.min(D, axis=1))
    dist = K.mean(tf.stack([dist_p_to_t, dist_t_to_p]))
    return dist


def chamfer_loss(y_true, y_pred):
    batched_losses = tf.map_fn(
        lambda x: chamfer_distance(x[0], x[1]), (y_true, y_pred), dtype=tf.float32
    )
    return K.mean(tf.stack(batched_losses))


def chamfer_loss_batch(y_true, y_pred):
    if K.int_shape(y_pred)[2] == 4:
        y_pred = y_pred[:, :, :-1]
        y_true = y_true[:, :, :-1]

    row_norms_true = tf.reduce_sum(tf.square(y_true), axis=2)
    row_norms_true = tf.reshape(row_norms_true, [tf.shape(y_pred)[0], -1, 1])

    row_norms_pred = tf.reduce_sum(tf.square(y_pred), axis=2)
    row_norms_pred = tf.reshape(row_norms_pred, [tf.shape(y_pred)[0], 1, -1])

    D = row_norms_true - 2 * tf.matmul(y_true, y_pred, False, True) + row_norms_pred

    dist_t_to_p = K.mean(K.min(D, axis=1))
    dist_p_to_t = K.mean(K.min(D, axis=2))
    dist = K.mean(tf.stack([dist_p_to_t, dist_t_to_p]))

    return dist


def gmm_nll_loss(covariance_matrix_diag, mix_param_val):
    def gmm_nll_batched(y_true, y_pred):
        def gmm_nll(y_true, y_pred):
            if K.int_shape(y_pred)[1] == 4:
                y_pred = y_pred[:, :-1]
                y_true = y_true[:, :-1]

            mix_param = tf.constant(mix_param_val)

            N = K.int_shape(y_pred)[0]
            uniform_pred = tf.constant(1 / N)
            point_prob = tf.fill([N], 1 / N)
            covariance_matrix = tf.constant(
                np.diag(
                    [
                        covariance_matrix_diag,
                        covariance_matrix_diag,
                        covariance_matrix_diag,
                    ]
                ),
                dtype=tf.float32,
            )

            mix_gauss_pred = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=point_prob),
                components_distribution=tfd.MultivariateNormalFullCovariance(
                    loc=y_pred, covariance_matrix=covariance_matrix
                ),
            )

            pdf = mix_gauss_pred.prob(y_true)
            pdf = (mix_param * uniform_pred) + ((1 - mix_param) * pdf)

            return -K.mean(tf.math.log(pdf))

        batched_losses = tf.map_fn(
            lambda x: gmm_nll(x[0], x[1]), (y_true, y_pred), dtype=tf.float32
        )
        return K.mean(batched_losses)

    return gmm_nll_batched
