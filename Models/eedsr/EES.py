from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, Conv2DTranspose
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import prepare_data as pd
import pandas
import numpy
import cv2


scale = 2

def model_EES():
    _input = Input(shape=(None, None, 1), name='input')

    EES = Conv2D(filters=4, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(_input)
    EES = Conv2DTranspose(filters=8, kernel_size=(14, 14), strides=(2, 2), padding='same', activation='relu')(EES)
    out = Conv2D(filters=1, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same')(EES)

    model = Model(_input, out)

    return model