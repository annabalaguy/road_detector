# ---Importation des données---


# defining function for dataLoading function
framObjTrain = {'img' : [],
           'mask' : []
          }

def LoadData( frameObj = None, imgPath = None, maskPath = None, shape = 128):
    imgNames = os.listdir(imgPath)
    maskNames = []

    ## generating mask names
    for mem in imgNames:
        mem = mem.split('_')[0]
        if mem not in maskNames:
            maskNames.append(mem)

    imgAddr = imgPath + '/'
    maskAddr = maskPath + '/'

    for i in range (len(imgNames)):
        try:
            img = plt.imread(imgAddr + maskNames[i] + '_sat.jpg')
            mask = plt.imread(maskAddr + maskNames[i] + '_mask.png')

        except:
            continue
        img = cv2.resize(img, (shape, shape))
        mask = cv2.resize(mask, (shape, shape))
        frameObj['img'].append(img)
        frameObj['mask'].append(mask[:,:,0]) # this is because its a binary mask and img is present in channel 0

    return frameObj

# ---Connexion au drive où les données sont stockées---

#mount drive
from google.colab import drive;
drive.mount('/content/drive')


#Definition du chemin d'accès pour importer les données
path='/content/drive/MyDrive/Colab_Notebooks/TropicalForestRoadDetectors/Kaggle/Dataset_1_Deep_globe/train/'
images_list = os.listdir(path) #Extract the name of the images
images_dict = {}

# --- Extraction des images, redimension et ajout de ces données dans des dictionnaires
nb_images=1200 #Nombre d'images à charger variable

for image in images_list[:nb_images] :
  Myimage = Image.open(path+image)
  Myimage = Myimage.resize((128,128),Image.ANTIALIAS)
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

#Reduction de la dimension du mask
def dim_reduction(list_mask):
    dimension = list_mask[0].shape[0]
    new_image = []

    for image in list_mask:
        a_img = []
        for a in range(dimension):
          b_img = []
          for b in range(dimension):
            print(b)
            b_img.append(image[a][b][0])
          a_img.append(b_img)
        new_image.append(a_img)

    new_image = np.array(new_image)
    return new_image

#Attribution des images test et des images train
TestObjTrain = {'img':np.array(dir_img), 'mask':dir_mask}

#TrainTestSplit
X_train, X_test, y_train, y_test = train_test_split(TestObjTrain['img'], TestObjTrain['mask'], test_size=0.3, random_state=1)


# --- Reshape des datasets---


#RESHAPE Y
y_train = y_train.reshape((420, 128, 128, 1))
y_test = y_test.reshape((180, 128, 128, 1))

#PAD X POUR GARDER BONNES DIM
X_train_pad=pad_sequences(X_train)
X_test_pad=pad_sequences(X_test)
