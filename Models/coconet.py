
# Math
import numpy as np
from math import atan2

# Machine Learning
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.optimizers import Adam

# Image Processing
#from skimage.measure import compare_psnr, compare_ssim, compare_mse
import cv2
import matplotlib.pyplot as plt

# Others
import os

class COCONET():
  def __init__(self, image_path, comp_path, decomp_path):
    self.image_path = image_path
    self.comp_path = comp_path
    self.decomp_path = decomp_path
    self.model = None

  def generate_placeholder_tensor(self, picture_sizex, picture_sizey, enhance = 1, trimensional = False):
    # Generate placeholder matrix with given dimensions
    X = []
    for x_it in range(0, picture_sizex * enhance):
        for y_it in range(0, picture_sizey * enhance):
            x0 = x_it / enhance + 0.5
            y0 = y_it / enhance + 0.5
            x = (x0 - picture_sizex / 2)
            y = (y0 - picture_sizey / 2)
            X.append((x0, y0, picture_sizex - x0, picture_sizey - y0, (x**2+y**2)**(1/2), atan2(y0, x0)))
    if (trimensional == False):
        return np.asarray(X)
    else:
        return np.reshape(np.asarray(X), (1, picture_sizex * enhance, picture_sizey * enhance, 6))

  def generate_value_tensor(self, img, picture_sizex, picture_sizey, trimensional = False):
    # Generate value matrix from image
    Y = []
    for x_iterator in range(0, picture_sizex):
        for y_iterator in range(0, picture_sizey):
            Y.append(np.multiply(1/255, img[x_iterator][y_iterator]))
    if (trimensional == False):
        return np.asarray(Y)
    else:
        return np.reshape(np.asarray(Y), (1, picture_sizex, picture_sizey, 3))

  def generate_model_dense(self, width_list):
    # Generate dense sequential model with fixed input and output and hidden layer widths from width_list
    model = Sequential()
    model.add(Dense(width_list[0], input_dim=6, activation = 'tanh', kernel_initializer = 'random_uniform'))
    for i in range(1, len(width_list)):
        model.add(Dense(width_list[i], activation = 'tanh'))
    model.add(Dense(3, activation = 'sigmoid'))
    model.compile(loss = 'mean_squared_error', optimizer = Adam(lr=0.001), metrics = ['accuracy'])
    model.save_weights('initial_weights.h5')
    return model

  def generate_model_conv(self, filters_list, dim = 10, slen = 1):
    # Generate conv sequential model with fixed input and output and filter counts from filters_list
    model = Sequential()
    model.add(Conv2D(kernel_size = (dim, dim), strides = slen, filters = filters_list[0], padding = 'same', input_shape=(None, None, 6), activation = 'tanh', kernel_initializer = 'random_uniform'))
    for i in range(1, len(filters_list)):
        model.add(Conv2D(kernel_size = (dim, dim), strides = slen, filters = filters_list[i], padding = 'same', activation = 'tanh'))
    model.add(Conv2D(kernel_size = (dim, dim), strides = slen, filters = 3, padding = 'same', activation = 'sigmoid'))
    model.compile(loss = 'mean_squared_error', optimizer = Adam(lr=0.0001), metrics = ['accuracy'])
    model.save_weights('initial_weights.h5')
    return model

  def load_image(self, address):
    # Load image as np.array and extract filename
    filename = os.path.basename(address)
    img = cv2.imread(address)
    return img, filename

  def compare_images(self, img1, img2):
    # Compute PSNR, SSIM and MSE for 2 images
    psnr = compare_psnr(img1, img2)
    ssim = compare_ssim(img1, img2, multichannel=True)
    mse = compare_mse(img1, img2)
    return psnr, ssim, mse

  def predict(self, model, X, picture_sizex, picture_sizey):
    # Predict
    prediction = model.predict(X)
    prediction = np.multiply(255, prediction)
    prediction = prediction.reshape(picture_sizex, picture_sizey, 3)
    return prediction.astype('uint8')

  def save_image(self, img, address):
    cv2.imwrite(address, img)

  def test_model(self):
    img, filename = self.load_image(self.image_path)
    picture_sizex = img.shape[0]
    picture_sizey = img.shape[1]
    enhance = 4

    X = self.generate_placeholder_tensor(picture_sizex, picture_sizey)
    X_SR = self.generate_placeholder_tensor(picture_sizex, picture_sizey, enhance = enhance)
    Y = self.generate_value_tensor(img, picture_sizex, picture_sizey)
    model = self.generate_model_dense([100] * 10)


    history = model.fit(X, Y, epochs = 1000, batch_size = 128, shuffle = True, verbose=1)
    #history = model.fit(X, Y, epochs = 1000, batch_size = 1024)
    prediction = self.predict(model, X_SR, picture_sizex * enhance, picture_sizey * enhance)

    plt.subplot(121)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.subplot(122)
    plt.imshow(cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB))
    plt.show()
    model.save_weights(self.comp_path)
    self.save_image(prediction, self.decomp_path)
        
