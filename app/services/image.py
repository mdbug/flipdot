from PIL import Image
import numpy as np

def load(file):
    image = Image.open("imgs/" + file)
    return np.asarray(image)


def crop(image: np.array) -> np.array:
    if image.shape[1] > image.shape[0]:
        crop_size = image.shape[0]
        start_x = (image.shape[1] - crop_size) // 2
        croped_image = image[:, start_x:start_x + crop_size]
    elif image.shape[0] > image.shape[1]:
        crop_size = image.shape[1]
        start_y = (image.shape[0] - crop_size) // 2
        croped_image = image[start_y:start_y + crop_size, :]
    else:
        croped_image = image
    
    return croped_image