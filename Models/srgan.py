import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import h5py
import cv2
from PIL import Image
import os

class VGG_LOSS(object):

  def __init__(self, image_shape):
    self.image_shape = image_shape
  
  def vgg_loss(self, y_true, y_pred):
    vgg19 = tf. VGG19(include_top=False, weights='imagenet', input_shape=self.image_shape)
    vgg19.trainable = False
    for l in vgg19.layers:
      l.trainable = False
    model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
    model.trainable = False
    return K.mean(K.square(model(y_true) - model(y_pred)))


class SRGAN():
  def __init__(self, image_path, output_path):
    self.image_path = image_path
    self.output_path = output_path
    self.model = None

  def normalize(self, input_data):
    return (input_data.astype(np.float32) - 127.5)/127.5 
  
  def denormalize(self, input_data):
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8)

  def load_model(self):
    tf.keras.losses.vgg_loss = VGG_LOSS.vgg_loss
    srgan = tf.keras.models.load_model('./Models/trained/srgan.h5', custom_objects={'vgg_loss': VGG_LOSS.vgg_loss})
    self.model=srgan
    return srgan
  
  def test_model(self):
    model = self.load_model()
    x_test_lr = self.load_test_data()
    examples = x_test_lr.shape[0]
    lr_img = self.denormalize(x_test_lr)
    gen_img = model.predict(x_test_lr)
    hr_img = self.denormalize(gen_img)

    plt.imsave(self.output_path,hr_img[0])
    plt.figure(figsize=[20,8])
    plt.subplot(1,2,1)
    plt.imshow(lr_img[0])
    plt.subplot(1,2,2)
    plt.imshow(hr_img[0])


  def load_test_data(self):
    path = self.image_path.split('.')
    if(path[-1] == 'png'):
      im1 = Image.open(self.image_path)
      im1.save('.'+path[1]+'.jpg')
      img = plt.imread('.'+path[1]+'.jpg')
      os.remove('.'+path[1]+'.jpg')
    else:
      img = plt.imread(self.image_path)
    # plt.imshow(img)
    x_test_lr = np.array([img])
    x_test_lr = self.normalize(x_test_lr)
    
    return x_test_lr


