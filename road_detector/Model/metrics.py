# --- IMPORTS ---

import tensorflow as tf
import tensorflow_addons
from tensorflow_addons import image
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


#custom metric with gaussian filter

def bruno_metric(y_true, y_pred):
    y_pred = binarization(y_pred, 0.001)

    y_true = tf.cast(tf.convert_to_tensor(y_true), "float")
    y_pred = tf.cast(tf.convert_to_tensor(y_pred), "float")

    y_true = gaussian_filter2d(y_true, sigma=4)
    y_pred = gaussian_filter2d(y_pred, sigma=4)

    y_true = tf.constant(y_true, dtype='float')
    y_pred = tf.constant(y_pred, dtype='float')
    #diviser par union: somme des 1 dans le pred et dans le true
    return tf.reduce_sum(tf.math.abs(y_true - y_pred)) / (len(y_true) *
                                                          (128 * 128))


#MSE with gaussian filter
def custom_mse(y_true, y_pred):
    #binarize
    by_pred = binarization(y_pred, 0.001)

    #gaussian pred
    gf_pred = gaussian_filter2d(by_pred, sigma=4)
    gf_true = gaussian_filter2d(y_true, sigma=4)
    # gf_true = y_true

    loss = metrics.mean_squared_error(y_true=gf_true, y_pred=gf_pred)
    return loss


#MAE with gaussian filter
def custom_mae(y_true, y_pred):
    #binarize
    by_pred = binarization(y_pred, 0.001)

    #gaussian blur
    gf_pred = gaussian_filter2d(by_pred, sigma=4)
    gf_true = gaussian_filter2d(y_true, sigma=4)

    loss = metrics.mean_absolute_error(y_true=gf_true, y_pred=gf_pred)
    return loss
