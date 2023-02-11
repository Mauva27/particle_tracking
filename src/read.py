import numpy as np
import cv2
from PIL import Image

def load_image(directory:str, filename:str, frame:int, format:str):
    '''
    imports images with specific format and returns arrays
    '''
    img = Image.open(f'{directory}{filename}{frame:05d}{format}',mode = 'r')
    return np.array(img.copy())

def load_movie_frames(directory,filename,frame,format):
    """
    returns the frame in movie as array
    """
    vidcap = cv2.VideoCapture(directory+filename+format)
    success, count = 1, 0
    while success:
        success, image = vidcap.read()
        if (count == frame) and success:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        count += 1
    return None
