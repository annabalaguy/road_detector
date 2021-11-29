# --- Imports ---

import tensorflow as tf
import tensorflow_addons
from tensorflow_addons import image
from tensorflow_addons.image import gaussian_filter2d
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import losses

def binarize(array, threshold):
    tmp = tf.convert_to_tensor(array)
    tmp = tf.cast(tf.math.greater(tmp, threshold), 'float32')
    return tmp

# ---LOSSES---

#Binary crossentropy with gaussian filter
def binary_crossentropy_gaussian(y_true, y_pred):
    #binarize
    bin_pred = binarize(y_pred, .5)

    #blur
    gf_pred = gaussian_filter2d(bin_pred, sigma=4)
    gf_true = gaussian_filter2d(y_true, sigma=4)

    #calc the loss
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    loss = bce(gf_true, gf_pred)
    return loss
