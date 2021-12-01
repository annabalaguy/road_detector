# --- Imports ---

import math
import random
import numpy as np
import tensorflow as tf
import sys

sys.path.append("data.py")
from data import y_test, y_train

#Création de l'IoU custom pour calculer manuellement la baseline

def iou(y_true, y_pred):
    '''
    IoU_loss
    '''
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    # y_pred = tf.math.sigmoid(y_pred)
    numerator = tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred) - numerator
    iou1 = numerator / denominator

    return tf.cast(iou1, tf.float32)


def baseline_generation(resolution):

    #The goal is to create a matrix with the shape of the images' dimensions filled with as many ones as there is on average in the y_train
    pixel_count = round(y_train.sum() / (y_train.shape[0] * y_train.shape[1] * y_train.shape[2]) * (resolution*resolution))
    base_array = []

    #Starts by creating a list of the correct number of ones
    for k in range(pixel_count):
        base_array.append(1)

    #Then add the zeros
    for a in range((resolution*resolution) - pixel_count):
        base_array.append(0)

    #Convert to array, randomize the distribution of ones and convert to correct dimension
    base_array = np.array(base_array)
    np.random.shuffle(base_array)
    base_array = base_array.reshape(resolution, resolution)

    return base_array

#y_train et y_test issus du code data.py

#testing baseline
num_prediction = 180
y_true = y_test
list_iou = []
list_mae = []
for i in range(len(y_true)):
    list_iou.append(float(iou(y_true[i], baseline_generation(256).reshape(256, 256, 1))))

print("Baseline IoU: ", np.mean(list_iou))
# print("Baseline MAE: ", np.mean(list_mae))
