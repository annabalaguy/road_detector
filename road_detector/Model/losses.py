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
    gf_true = tf.constant(y_true, dtype='float')

    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    loss = bce(gf_true, gf_pred)
    return loss
