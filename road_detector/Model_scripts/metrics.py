# --- IMPORTS ---

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow_addons.image import gaussian_filter2d
from tensorflow.keras import metrics

#Definition de la binarisation pour le calcul des métriques
def binarization(y_pred, threshold):
    y_pred = y_pred.numpy()
    print(type(y_pred))
    print(y_pred.shape)
    batch_size = y_pred.shape[0]
    thresh = threshold
    binarized = []

    for image in y_pred:
        row = []
        for row in image:
            col = []
            for col in row:
                value = y_pred[row][col]
                if value > threshold:
                    value = 1
                else:
                    value = 0
                col.append(value)
            row.append(col)
        binarized.append(row)

    return tf.constant(binarized, dtype='int')


# --- METRICS ---


#Vincent metrics
def binarize(array, threshold):
    tmp = tf.convert_to_tensor(array)
    tmp = tf.cast(tf.math.greater(tmp, threshold), 'float32')
    return tmp


#continuous_iou
def metrics_continuous_iou(y_true, y_pred, sigma=(4.0, 4.0)):
    y_true = tf.cast(tf.convert_to_tensor(y_true), 'float32')
    y_pred = tf.cast(tf.convert_to_tensor(y_pred), 'float32')
    bin_pred = binarize(y_pred, .5)
    gf_true = tfa.image.gaussian_filter2d(y_true, sigma=sigma)
    gf_pred = tfa.image.gaussian_filter2d(bin_pred, sigma=sigma)
    y_true = tf.cast(gf_true, dtype='float32')
    y_pred = tf.cast(gf_pred, dtype='float32')
    return tf.math.reduce_sum(tf.math.abs(tf.math.subtract(
        gf_true, gf_pred))) / tf.math.reduce_sum(y_true + bin_pred)


#MSE with gaussian filter
def custom_mse(y_true, y_pred):
    gf_pred = gaussian_filter2d(y_pred, sigma=4)
    gf_true = gaussian_filter2d(y_true, sigma=4)
    # gf_true = y_true

    loss = metrics.mean_squared_error(y_true=gf_true, y_pred=gf_pred)
    return loss


#MAE with gaussian filter
def custom_mae(y_true, y_pred):

    gf_pred = gaussian_filter2d(y_pred, sigma=4)
    gf_true = gaussian_filter2d(y_true, sigma=4)
    # gf_true = y_true

    loss = metrics.mean_absolute_error(y_true=gf_true, y_pred=gf_pred)
    return loss
