#%%
# Math
import numpy as np
from math import atan2

# Machine Learning
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D

# Image Processing
#from skimage.measure import compare_psnr, compare_ssim, compare_mse
import cv2
import matplotlib.pyplot as plt

# Others
import os

def generate_placeholder_tensor(picture_sizex, picture_sizey, enhance = 1, trimensional = False):
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

def generate_value_tensor(img, picture_sizex, picture_sizey, trimensional = False):
    # Generate value matrix from image
    Y = []
    for x_iterator in range(0, picture_sizex):
        for y_iterator in range(0, picture_sizey):
            Y.append(np.multiply(1/255, img[x_iterator][y_iterator]))
    if (trimensional == False):
        return np.asarray(Y)
    else:
        return np.reshape(np.asarray(Y), (1, picture_sizex, picture_sizey, 3))

def generate_model_dense(width_list):
    # Generate dense sequential model with fixed input and output and hidden layer widths from width_list
    model = Sequential()
    model.add(Dense(width_list[0], input_dim=6, activation = 'tanh'))
    for i in range(1, len(width_list)):
        model.add(Dense(width_list[i], activation = 'tanh'))
    model.add(Dense(3, activation = 'sigmoid'))
    model.compile(loss = 'mean_squared_error', optimizer = tensorflow.keras.optimizers.Adam(lr=0.001), metrics = ['accuracy'])
    model.save_weights('initial_weights.h5')
    return model

def generate_model_conv(filters_list, dim = 10, slen = 1):
    # Generate conv sequential model with fixed input and output and filter counts from filters_list
    model = Sequential()
    model.add(Conv2D(kernel_size = (dim, dim), strides = slen, filters = filters_list[0], padding = 'same', input_shape=(None, None, 6), activation = 'tanh'))
    for i in range(1, len(filters_list)):
        model.add(Conv2D(kernel_size = (dim, dim), strides = slen, filters = filters_list[i], padding = 'same', activation = 'tanh'))
    model.add(Conv2D(kernel_size = (dim, dim), strides = slen, filters = 3, padding = 'same', activation = 'sigmoid'))
    model.compile(loss = 'mean_squared_error', optimizer = tensorflow.keras.optimizers.Adam(lr=0.0001), metrics = ['accuracy'])
    model.save_weights('initial_weights.h5')
    return model

def load_image(address):
    # Load image as np.array and extract filename
    filename = os.path.basename(address)
    img = cv2.imread(address)
    return img, filename

def compare_images(img1, img2):
    # Compute PSNR, SSIM and MSE for 2 images
    psnr = compare_psnr(img1, img2)
    ssim = compare_ssim(img1, img2, multichannel=True)
    mse = compare_mse(img1, img2)
    return psnr, ssim, mse

def predict(model, X, picture_sizex, picture_sizey):
    # Predict
    prediction = model.predict(X)
    prediction = np.multiply(255, prediction)
    prediction = prediction.reshape(picture_sizex, picture_sizey, 3)
    return prediction.astype('uint8')

def save_image(img, address):
    cv2.imwrite(address, img)

#%%
picture_sizex = 32
picture_sizey = 32
#enhance = 1
enhance = 4

path = input('Image file: ')

#%%
img, filename = load_image(path)
plt.imshow(img)
#%%
X = generate_placeholder_tensor(picture_sizex, picture_sizey)
X_SR = generate_placeholder_tensor(picture_sizex, picture_sizey, enhance = enhance)
Y = generate_value_tensor(img, picture_sizex, picture_sizey)
model = generate_model_dense([100] * 10)

#%%
# history = model.fit(X, Y, epochs = 1000, batch_size = 128, shuffle = True)
#history = model.fit(X, Y, epochs = 1000, batch_size = 1024)
prediction = predict(model, X_SR, picture_sizex * enhance, picture_sizey * enhance)

plt.subplot(121)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(122)
plt.imshow(cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB))
# plt.show()

save_image(prediction, path[:-4] + ' FUSED.png')
# %%
