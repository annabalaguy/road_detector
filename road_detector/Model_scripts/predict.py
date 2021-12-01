
import numpy as np
from tensorflow.keras import models
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from model import GiveMeUnet
from data import X_test_pad, y_test

def predict16 (valMap, model, shape = 256):
    ## getting and proccessing val data
    test_shape = y_test.shape[0]
    img = valMap['img'][0:test_shape]
    #mask = valMap['mask'][0:test_shape]
    #mask = mask[0:50]

    imgProc = img [0:test_shape]
    imgProc = np.array(img)

    predictions = model.predict(imgProc)
    return predictions, imgProc #, mask

X='image'
#Importation du modèle
data_path='/Users/loulou/code/annabalaguy/road_detector/road_detector/BasicRouteGenerator.h5'
unet=GiveMeUnet(X)
models.load_model(data_path)

#Importation des données et preprocessing ?
X_pad=pad_sequences(X)

#prediction du modèle
unet.predict(X_pad, verbose=1)
