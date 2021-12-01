from fastapi import FastAPI
from tensorflow.keras import models
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from road_detector.Model_scripts.losses import binary_crossentropy_gaussian
from road_detector.Model_scripts.metrics import metrics_continuous_iou, custom_mae
import base64
import numpy as np
from pydantic import BaseModel
from road_detector.Model_scripts.helpers import make_image_api_friendly

#désinstaller environnement
#nouvel envmt virtuel
#Nvel pip install

app = FastAPI()
path_local_model='/Users/loulou/code/annabalaguy/road_detector/road_detector/BasicRouteGenerator.h5'
model = models.load_model(path_local_model,
                          custom_objects={
                              "binary_crossentropy_gaussian":
                              binary_crossentropy_gaussian,
                              "metrics_continuous_iou": metrics_continuous_iou, "custom_mae":custom_mae
                          })

#Upload de l'Image par l'utilisateur


class Item(BaseModel):
    image: str
    height: int
    width: int
    channel: int

@app.get("/")
def test():
    return "OK"

@app.post("/predict")
async def prediction(item:Item):
    # Decode received string from bytes
    bytes_string = bytes(item.image, 'utf-8')
    # Decode from b64
    decoded_string = base64.b64decode(bytes_string)
    # Convert to numpy array and force uint8 to match initial type
    img = np.frombuffer(decoded_string, dtype='uint8')
    # Reshape to original shape
    img = img.reshape((item.height, item.width, item.channel))

    # possibilité de faire preproc + preidct à cet endroit
    img=pad_sequences(img)
    predict_img=model.predict(img)

    image_predite=make_image_api_friendly(predict_img)

    return image_predite

#prendre l'image en str puis la transformer en ?? grâce à PIL, puis la transformer en array puis liste pour lecture modele


@app.get("/test")
def home():
    return {"Bonjour à tous"}


@app.get("/")
def index():
    return {"Bonjour à tous"}
