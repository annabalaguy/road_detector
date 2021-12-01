# --- Imports ---

import sys
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from model import GiveMeUnet
from losses import binary_crossentropy_gaussian
from metrics import custom_mse, custom_mae, metrics_continuous_iou
from tensorflow.keras.optimizers.schedules import ExponentialDecay

sys.path.append("data.py")
from data import y_test, y_train, X_test_pad, X_train_pad

lr_schedule = ExponentialDecay(
    initial_learning_rate=0.001, decay_steps=5000, decay_rate=0.7)

## instantiating model
inputs = tf.keras.layers.Input((256, 256, 3))

unet = GiveMeUnet(inputs, droupouts= 0.07)
adam = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
unet.compile(optimizer = adam, loss = binary_crossentropy_gaussian, metrics=[metrics_continuous_iou, custom_mae], run_eagerly=True)


#Fit and train the model

es = EarlyStopping(patience=20, restore_best_weights=True)

retVal = unet.fit(X_train_pad, y_train,
                  validation_split=0.2,
                  epochs = 200,
                  verbose = 1,
                  batch_size=16,
                  callbacks=[es])


unet.evaluate(X_test_pad, y_test, verbose=1)


#PLOT HISTORY ??
#Importer modèle avec joblib ??
