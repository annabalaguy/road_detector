# --- Imports ---

import tensorflow as tf
import tensorflow_addons
from tensorflow_addons import image
from tensorflow_addons.image import gaussian_filter2d
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import losses

# ---LOSSES---


#Binary crossentropy with gaussian filter
def binary_crossentropy_gaussian(y_true, y_pred):
    gf_pred = gaussian_filter2d(y_pred, sigma=4)
    gf_true = tf.constant(y_true, dtype='int64')

    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    loss = bce(gf_true, gf_pred)
    return loss


#New custom loss
def jaccard_distance(y_true, y_pred, smooth=100):
    y_true = tf.cast(y_true, 'float32')
    y_pred_bin = tf.round(y_pred)
    y_pred = (y_pred - (tf.stop_gradient(y_pred) - y_pred_bin))
    intersection = tf.reduce_sum(tf.abs(y_true) * tf.abs(y_pred))
    sum_ = tf.reduce_sum(tf.abs(y_true) + tf.abs(y_pred))
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth
