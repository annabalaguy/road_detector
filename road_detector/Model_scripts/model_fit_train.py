# --- Imports ---

import sys
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from model import GiveMeUnet
from losses import binary_crossentropy_gaussian
from metrics import custom_mse, custom_mae, metrics_continuous_iou

sys.path.append("data.py")
from data import y_test, y_train, X_test_pad, X_train_pad

## instantiating model
inputs = tf.keras.layers.Input((128, 128, 3))

unet = GiveMeUnet(inputs, droupouts= 0.07)

#optimisation du Adam
adam = tf.keras.optimizers.Adam(learning_rate=0.001)
#Compilation
unet.compile(optimizer = adam, loss = binary_crossentropy_gaussian, metrics=[custom_mse, custom_mae], run_eagerly=True)

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
