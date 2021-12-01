from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from road_detector.Model_scripts.data import get_data
from road_detector.Model_scripts.model import GiveMeUnet
from termcolor import colored
import numpy as np
import pdb
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import losses
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import tensorflow_addons as tfa
from tensorflow_addons import image
from tensorflow_addons.image import gaussian_filter2d
from tensorflow.keras.preprocessing.sequence import pad_sequences
from road_detector.Model_scripts.losses import binary_crossentropy_gaussian
from road_detector.Model_scripts.metrics import metrics_continuous_iou, custom_mae
from google.cloud import storage

BUCKET_NAME = "wagon-data-722-road-detector"


class Trainer:
    def __init__(self):
        pass

    def give_unet(self):
        lr_schedule = ExponentialDecay(
        initial_learning_rate=0.001, decay_steps=5000, decay_rate=0.7)
        ## instantiating model
        inputs = tf.keras.layers.Input((256, 256, 3))

        unet = GiveMeUnet(inputs, droupouts= 0.07)
        adam = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        unet.compile(optimizer = adam, loss = binary_crossentropy_gaussian, metrics=[metrics_continuous_iou, custom_mae], run_eagerly=True)
        self.unet = unet

    def run(self):
        es = EarlyStopping(patience=20, restore_best_weights=True)

        self.unet.fit(X_train_pad, y_train,
        validation_split=0.2,
        epochs = 1,
        verbose = 1,
        batch_size=16,
        callbacks=[es])


    def evaluate(self, X_test_pad, y_test):
        return self.unet.evaluate(X_test_pad, y_test, verbose=1)

    # def save_model(self):
    #     """Save the model into a .joblib format"""
    #     unet.save(
    #         'jaccard_RouteGenerator.h5'
    #     )
    #     unet.save_weights(
    #         'jaccard_RouteGenerator_weights.h5'
    #     )
    #     print(colored("h5 and weights saved locally", "green"))


    def save_model_to_gcp(self):
        """Save the model into a .joblib and upload it on Google Storage /models folder
        HINTS : use sklearn.joblib (or jbolib) libraries and google-cloud-storage"""
        local_model_name = 'jaccard_RouteGenerator.h5'
        local_weight_name = 'jaccard_RouteGenerator_weights.h5'
        self.unet.save(local_model_name)
        self.unet.save_weights(local_weight_name)
        client = storage.Client().bucket(BUCKET_NAME)
        storage_location_model = f"Models/{local_model_name}"
        storage_location_weight = f"Models/{local_weight_name}"
        blob_model = client.blob(storage_location_model)
        blob_weight = client.blob(storage_location_weight)
        blob_model.upload_from_filename(local_model_name)
        blob_weight.upload_from_filename(local_weight_name)
        print("uploaded h5 to gcp cloud storage under \n => {}".format(
            storage_location_model))
        print("uploaded h5 weights to gcp cloud storage under \n => {}".format(
            storage_location_weight))




if __name__ == "__main__":
    # Get and clean data
    dir_img, dir_mask = get_data()
    # ⚠️ alternatively use data from gcp with get_data_from_gcp
    #TestObjTrain = {'img': np.array(dir_img), 'mask': dir_mask}
    dir_img, dir_mask = np.array(dir_img), np.array(dir_mask)
    X_train, X_test, y_train, y_test = train_test_split(
        dir_img, dir_mask, test_size=0.2,
        random_state=1)  # Train and save model, locally and
    pad=pad_sequences(X_train)
    #Adding to the black and white masks a dimension
    y_train = y_train.reshape((X_train.shape[0], 256, 256, 1))
    y_test = y_test.reshape((X_test.shape[0], 256, 256, 1))
    #PAD X POUR GARDER BONNES DIM
    X_train_pad=pad_sequences(X_train)
    X_test_pad=pad_sequences(X_test)

    trainer = Trainer()
    unet = trainer.give_unet()
    trainer.run(X_train_pad, y_train)
    loss_score = trainer.evaluate(X_test, y_test)
    print(f"loss_score: {loss_score}")
    trainer.save_model_to_gcp()
