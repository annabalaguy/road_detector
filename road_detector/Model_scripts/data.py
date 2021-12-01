# --- Importation modules ---
import os
import numpy as np
import pandas as pd
import IPython.display as display
from PIL import Image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from google.cloud import storage


# --- IMPORTATION DES DONNEES ---
# ---Connexion à GCP---

# project id
PROJECT_ID = "wagon-328013"

# bucket name
BUCKET_NAME = "wagon-data-722-road-detector"

#Definition du chemin d'accès pour importer les données

images_dict = {}


#@simple_time_tracker
def get_data_from_gcp(nrows=12453, **kwargs):
    """method to get the training data (or a portion of it) from google cloud bucket"""
    # Images' extraction
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blobs = bucket.list_blobs(prefix="Data/train/100034")
    for blob in blobs :
        if os.path.basename(blob.name) == "":
            continue
        blob.download_to_filename(os.path.basename(blob.name))
    #Creation of a dictionnary
    nb_image = nrows
    images_list = os.listdir()
    for image in images_list[:nb_image]:
        Myimage = Image.open(image)
        #Resizing
        Myimage = Myimage.resize((256, 256), Image.ANTIALIAS)
        #Binarizing the mask images
        if 'mask' in image :
            thresh = 128
            fn = lambda x : 255 if x > thresh else 0
            Myimage = Myimage.convert(mode='L').point(fn, mode='1')
        #Converting in arrays
        image_array = np.array(Myimage)
        #Creation of the dictionnary
        images_dict[image] = image_array

    #create lists of the masks and raw images array
    dir_mask = []
    dir_img = []

    for key in images_dict.keys():
        if 'mask' in str(key):
            dir_mask.append(images_dict[key])
        if 'sat' in str(key):
            dir_img.append(images_dict[key])

    #Transformation du type des données de masque de Booléen "True", "False" en 0, 1
    dir_mask = np.array(dir_mask).astype('int')
    return dir_img, dir_mask


if __name__ == '__main__':
    dir_img, dir_mask = get_data_from_gcp()
