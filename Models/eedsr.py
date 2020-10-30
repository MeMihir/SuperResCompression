from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, add, Conv2D, Input, Conv2DTranspose, UpSampling2D, concatenate, Activation
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint
# import prepare_data as pd
# import pandas as pd
import numpy
import math
import cv2

scale = 2

class EEDSR():
  def __init__(self,image_path, output_path):
    self.image_path = image_path
    self.output_path = output_path
    self.model = None

  def model_EES(self):
    _input = Input(shape=(None, None, 1), name='input')
    EES = Conv2D(filters=4, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(_input)
    EES = Conv2DTranspose(filters=8, kernel_size=(14, 14), strides=(2, 2), padding='same', activation='relu')(EES)
    out = Conv2D(filters=1, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same')(EES)
    model = Model(_input, out)
    return model

  def Res_block(self):
    _input = Input(shape=(None, None, 64))

    conv = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(_input)
    conv = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='linear')(conv)

    out = add([_input, conv])
    out = Activation('relu')(out)

    model = Model(inputs=_input, outputs=out)

    return model


  def model_EED(self):
    _input = Input(shape=(None, None, 1), name='input')

    Feature = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(_input)
    Feature_out = self.Res_block()(Feature)

    # Upsampling
    Upsampling1 = Conv2D(4, (1, 1), strides=(1, 1), padding='same', activation='relu')(Feature_out)
    Upsampling2 = UpSampling2D((14, 14))(Upsampling1)
    Upsampling3 = Conv2D(64, (1, 1), strides=(1, 1), padding='same', activation='relu')(Upsampling2)

    # Mulyi-scale Reconstruction
    Reslayer1 = self.Res_block()(Upsampling3)

    Reslayer2 = self.Res_block()(Reslayer1)

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

    Multi_scale2 = concatenate(inputs=[Multi_scale2a, Multi_scale2b, Multi_scale2c, Multi_scale2d], axis=0)

    out = Conv2D(1, (1, 1), strides=(1, 1), padding='same', activation='relu')(Multi_scale2)
    model = Model(_input, out)

    return model

  def model_EEDS(self):
    _input = Input(shape=(None, None, 1), name='input')
    _EES = self.model_EES()(_input)
    _EED = self.model_EED()(_input)
    _EEDS = add([_EED, _EES])

    model = Model(_input, _EEDS)
    model.compile(optimizer=Adam(lr=0.0003), loss='mse')
    self.model = model
    model.load_weights("./")
    return model

  def EEDS_predict(self):
    INPUT_NAME = "input.jpg"

    label = cv2.imread(self.image_path)
    shape = label.shape

    img = cv2.resize(label, (int(shape[1] / scale), int(shape[0] / scale)), cv2.INTER_CUBIC)
    cv2.imwrite(INPUT_NAME, img)

    EEDS = self.model_EEDS()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y = numpy.zeros((1, img.shape[0], img.shape[1], 1))
    Y[0, :, :, 0] = img[:, :, 0].astype(float) / 255.
    img = cv2.cvtColor(label, cv2.COLOR_BGR2YCrCb)

    pre = EEDS.predict(Y, batch_size=1) * 255.
    pre[pre[:] > 255] = 255
    pre = numpy.uint8(pre)
    img[:, :, 0] = pre[0, :, :, 0]
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite('../', img)

    # psnr calculation:
    # im1 = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
    # im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2YCrCb)
    # im2 = cv2.imread(INPUT_NAME, cv2.IMREAD_COLOR)
    # im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2YCrCb)
    # im2 = cv2.resize(im2, (img.shape[1], img.shape[0]))
    # cv2.imwrite("Bicubic.jpg", cv2.cvtColor(im2, cv2.COLOR_YCrCb2BGR))
    # im3 = cv2.imread(OUTPUT_NAME, cv2.IMREAD_COLOR)
    # im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2YCrCb)

    # print("Bicubic:")
    # print(cv2.PSNR(im1[:, :, 0], im2[:, :, 0]))
    # print("EEDS:")
    # print(cv2.PSNR(im1[:, :, 0], im3[:, :, 0]))