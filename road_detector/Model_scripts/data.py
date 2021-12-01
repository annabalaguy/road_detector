# --- Importation modules ---
import os
import IPython.display as display
from PIL import Image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
import pdb

# --- IMPORTATION DES DONNEES ---
# ---Connexion au drive où les données sont stockées---

#Lecture des fichiers dans Google drive
#from google.colab import drive;
#drive.mount('/content/drive')

#Definition du chemin d'accès pour importer les données
path='/Users/Anna/code/annabalaguy/road_detector/100_images_train/'
images_list = os.listdir(path) #Extract the name of the images
images_dict = {}

def get_data():
    # --- Extraction des images, redimension et ajout de ces données dans des dictionnaires
    nb_images= 93 #Nombre d'images à charger variable
    for image in images_list[:nb_images] :
        Myimage = Image.open(path+image)
        Myimage = Myimage.resize((256,256),Image.ANTIALIAS)
        if 'mask' in image:
            thresh = 10
            fn = lambda x : 255 if x > thresh else 0
            Myimage = Myimage.convert(mode='L').point(fn, mode='1')
        image_array = np.array(Myimage)
        images_dict[image] = image_array
    #create directory for masks and raw images
    dir_mask = []
    dir_img = []

    for key in images_dict.keys():
        if 'mask' in str(key):
            dir_mask.append(images_dict[key])
        if 'sat' in str(key):
            dir_img.append(images_dict[key])

    #Transformation du type des données de masque de Booléen "True", "False" en 0, 1
    dir_mask = np.array(dir_mask).astype('int')
    dir_img = np.array(dir_img)


    return dir_img, dir_mask


if __name__ == "__main__":
    get_data()
