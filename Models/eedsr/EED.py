from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Input, UpSampling2D, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate, add
import prepare_data as pd
import numpy


def Res_block():
    _input = Input(shape=(None, None, 64))

    conv = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(_input)
    conv = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='linear')(conv)

    out = add([_input, conv])
    out = Activation('relu')(out)

    model = Model(inputs=_input, outputs=out)

    return model


def model_EED():
    _input = Input(shape=(None, None, 1), name='input')

    Feature = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(_input)
    Feature_out = Res_block()(Feature)

    # Upsampling
    Upsampling1 = Conv2D(4, (1, 1), strides=(1, 1), padding='same', activation='relu')(Feature_out)
    Upsampling2 = UpSampling2D((14, 14))(Upsampling1)
    Upsampling3 = Conv2D(64, (1, 1), strides=(1, 1), padding='same', activation='relu')(Upsampling2)

    # Mulyi-scale Reconstruction
    Reslayer1 = Res_block()(Upsampling3)

    Reslayer2 = Res_block()(Reslayer1)

    # ***************//
    Multi_scale1 = Conv2D(16, (1, 1), strides=(1, 1), padding='same', activation='relu')(Reslayer2)

    Multi_scale2a = Conv2D(16, (1, 1), strides=(1, 1),
                           padding='same', activation='relu')(Multi_scale1)

    Multi_scale2b = Conv2D(16, (1, 3), strides=(1, 1),
                           padding='same', activation='relu')(Multi_scale1)
    Multi_scale2b = Conv2D(16, (3, 1), strides=(1, 1),
                           padding='same', activation='relu')(Multi_scale2b)

    Multi_scale2c = Conv2D(16, (1, 5), strides=(1, 1),
                           padding='same', activation='relu')(Multi_scale1)
    Multi_scale2c = Conv2D(16, (5, 1), strides=(1, 1),
                           padding='same', activation='relu')(Multi_scale2c)

    Multi_scale2d = Conv2D(16, (1, 7), strides=(1, 1),
                           padding='same', activation='relu')(Multi_scale1)
    Multi_scale2d = Conv2D(16, (7, 1), strides=(1, 1),
                           padding='same', activation='relu')(Multi_scale2d)

    Multi_scale2 = concatenate(inputs=[Multi_scale2a, Multi_scale2b, Multi_scale2c, Multi_scale2d])

    out = Conv2D(1, (1, 1), strides=(1, 1), padding='same', activation='relu')(Multi_scale2)
    model = Model(_input, out)

    return model
