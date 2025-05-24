from PIL import Image
from numpy import asarray

def load_image(file):
    image = Image.open("imgs/" + file)
    return asarray(image)