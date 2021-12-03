import base64
import numpy as np
from PIL import Image


def image_to_dict(image_array, dtype='uint8', encoding='utf-8'):
    '''
    Convert an ndarray representing a batch of images into a compressed string

    ----------
    Parameters
    ----------
    imgArray: a np array representing an image

    ----------
    Returns
    ----------
    dict(image: str,
         height: int,
         width: int,
         channel: int)
    '''
    # Get array shape
    if image_array.ndim < 2 or image_array.ndim > 4:
        raise TypeError
    elif image_array.ndim < 3:
        size = 1
        channel = 1
        height, width = image_array.shape
    elif image_array.ndim < 4:
        size = 1
        height, width, channel = image_array.shape
    else:
        size, height, width, channel = image_array.shape

    # Ensure uint8
    image_array = image_array.astype(dtype)
    # Flatten image
    image_array = image_array.reshape(size * height * width * channel)
    # Encode in b64 for compression
    image_array = base64.b64encode(image_array)
    # Prepare image for POST request, ' cannot be serialized in json
    image_array = image_array.decode(encoding).replace("'", '"')

    api_dict = {'image': image_array, 'size': size, 'height': height,
                'width': width, 'channel': channel}

    return api_dict


def image_from_dict(api_dict, dtype='uint8', encoding='utf-8'):
    '''
    Convert an item representing a batch of images into a ndarray
    item is an instance of an Item class,
    inheriting from BaseModel pydantic class

    ----------
    Parameters
    ----------
    api_dict: an item(image, height, width, channel) representing an image
    dtype: target data type for ndarray
    encoding: encoding used for image string

    ----------
    Returns
    ----------
    ndarray of shape (size, height, width, channel)
    '''
    # Decode image string
    img = base64.b64decode(bytes(api_dict.image, encoding))
    # Convert to np.ndarray and ensure dtype
    img = np.frombuffer(img, dtype=dtype)
    # Reshape to original shape
    img = img.reshape((api_dict.size,
                       api_dict.height,
                       api_dict.width,
                       api_dict.channel))

    return img


def binarize_predictions(pred, threshold=0.5):
    #Binarize the prediction based on if above or below a given threshold (default = 0.5)
    dimension = pred.shape[0]
    new_pred = []
    #vectorize
    pred = list(pred.reshape(dimension * dimension))
    #binarize the vector
    for pixel in pred:
        if pixel > threshold:
            new_pred.append(255)
        else:
            new_pred.append(0)

    new_pred = np.array(new_pred).reshape(dimension, dimension)
    return new_pred


def display_resized_prediction(pred):
    #Resize the image in 526, 526 & display it
    prediction = binarize_predictions(pred)
    image = Image.fromarray((prediction * 255).astype(np.uint8))
    image = image.resize((526, 526))
    image = np.array(image)
    return image
