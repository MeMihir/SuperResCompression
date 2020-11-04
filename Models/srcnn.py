import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, Input, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam

class SRCNN():
  def __init__(self,image_path, output_path):
    self.image_path = image_path
    self.output_path = output_path
    self.model = None

  def load_model(self):
    srcnn = Sequential()
    srcnn.add(Conv2D(128, (9, 9), kernel_initializer='glorot_uniform',
                     activation='relu', padding='valid', use_bias=True, input_shape=(None, None, 1)))
    srcnn.add(Conv2D(64, (3, 3), kernel_initializer='glorot_uniform',
                     activation='relu', padding='same', use_bias=True))
    # srcnn.add(BatchNormalization())
    srcnn.add(Conv2D(1, (5, 5), kernel_initializer='glorot_uniform',
                     activation='linear', padding='valid', use_bias=True))
    adam = Adam(lr=0.0003)
    srcnn.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])  
    srcnn.load_weights("./Models/trained/srcnn.h5")
    self.model=srcnn
    return srcnn
  
  def predict(self):
    srcnn_model = self.load_model()
    INPUT_NAME = "input2.jpg"

    img = cv2.imread(self.image_path, cv2.IMREAD_COLOR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    shape = img.shape
    Y_img = img.copy()[:,:,0]
    # Y_img = cv2.resize(img[:, :, 0], (int(shape[1] / 2), int(shape[0] / 2)), cv2.INTER_CUBIC)
    # Y_img = cv2.resize(Y_img, (shape[1], shape[0]), cv2.INTER_CUBIC)
    img[:, :, 0] = Y_img
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    # cv2.imwrite(INPUT_NAME, img)

    Y = np.zeros((1, img.shape[0], img.shape[1], 1), dtype=float)
    Y[0, :, :, 0] = Y_img.astype(float) / 255.
    pre = srcnn_model.predict(Y, batch_size=1) * 255.
    pre[pre[:] > 255] = 255
    pre[pre[:] < 0] = 0
    pre = pre.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    img[6: -6, 6: -6, 0] = pre[0, :, :, 0]
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    
    plt.figure(figsize=[20,8])
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.subplot(1,2,2)
    orgimg = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
    plt.imshow(cv2.cvtColor(orgimg, cv2.COLOR_BGR2RGB))

    cv2.imwrite(self.output_path, img)

