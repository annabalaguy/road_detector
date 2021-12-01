import base64


def make_image_api_friendly(imgArray):
    '''
    Convert the ndarray representing an image into a compressed string
    ----------
    Parameters
    ----------
    imgArray: numpy array representing an image

    ----------
    Returns
    ----------
    api_dict: dictionary(image: str, height: int, width: int, channel: int)
    '''
    # Encode all image into 1 dim uint8 string
    # Ensure uint8
    img = imgArray.astype('uint8')
    # Get current shape, only for single image
    height, width, channel = img.shape
    # Flatten image
    img_reshape = img.reshape(height*width*channel)
    # Encode in b64 for compression
    img_enc = base64.b64encode(img_reshape)
    # Prepare image for POST request, ' cannot be serialized in json
    img_post = img_enc.decode('utf8').replace("'", '"')
    api_dict = {'image': img_post, 'height': height,
                'width': width, 'channel': channel}
    return api_dict
