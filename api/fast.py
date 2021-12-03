from fastapi import FastAPI
from tensorflow.keras import models
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import numpy as np
from pydantic import BaseModel
from api.helpers import image_from_dict, image_to_dict, binarize_predictions, display_resized_prediction
from road_detector.unet import GiveMeUnet
app = FastAPI()
LOCAL_Weights = 'WEIGHTS_Vincent_Halfdata_Crossentropy.h5'
#/Users/loulou/code/annabalaguy/road-api/road_detector/WEIGHTS_Vincent_Jaccard_Crossentropy.h5
unet = GiveMeUnet()
unet.load_weights(LOCAL_Weights)

#Upload de l'Image par l'utilisateur

class Item(BaseModel):
    image: str
    size : int
    height: int
    width: int
    channel: int

@app.get("/")
def test():
    return {"status": "OK"}

@app.post("/predict")
async def prediction(item:Item):
    #Conversion de l'image en nparray
    img = image_from_dict(item, dtype='float32')

    # Preproc + predict à cet endroit
    #img=pad_sequences(img)

    #Prediction de l'image

    predict_img=unet.predict(img)

    #Conversion du nparray en image str lisible pour l'API

    #pred_binary_img=binarize_predictions(predict_img)
    #pred_resized_img=display_resized_prediction(pred_binary_img)

    image_predite = image_to_dict(predict_img, dtype='float16')


    # TO DO: Option to convert tf to np -> prediction.numpy()
    # TO DO: Option to unscale *255 -> prediction * 255

    return image_predite


@app.get("/test")
def home():
    return {"Bonjour à tous"}


@app.get("/")
def index():
    return {"Bonjour à tous"}
