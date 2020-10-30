import tensorflow as tf
import numpy as np

class SRCNN():
  def __init__(self,image_path, output_path):
    self.image_path = image_path
    self.output_path = output_path
    self.model = None

  def load_model(self):
    srcnn = tf.keras.models.Sequential()
    srcnn.add(tf.keras.layers.Conv2D(128, (9, 9),
                    activation='relu', padding='valid', input_shape=(None, None, 1)))
    srcnn.add(tf.keras.layers.Conv2D(64, (3, 3),
                    activation='relu', padding='same'))
    srcnn.add(tf.keras.layers.Conv2D(1, (5, 5),
                    activation='linear', padding='valid'))
    adam = tf.keras.optimizers.Adam(lr=0.0003)
    srcnn.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
  
    srcnn.load_weights("./Models/trained/srcnn.h5")
    self.model=srcnn
    return srcnn
  
  def predict(self):
    srcnn_model = self.load_model()
    INPUT_NAME = "input2.jpg"

    import cv2
    img = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    shape = img.shape
    Y_img = cv2.resize(img[:, :, 0], (int(shape[1] / 2), int(shape[0] / 2)), cv2.INTER_CUBIC)
    Y_img = cv2.resize(Y_img, (shape[1], shape[0]), cv2.INTER_CUBIC)
    img[:, :, 0] = Y_img
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(INPUT_NAME, img)

    Y = np.zeros((1, img.shape[0], img.shape[1], 1), dtype=float)
    Y[0, :, :, 0] = Y_img.astype(float) / 255.
    pre = srcnn_model.predict(Y, batch_size=1) * 255.
    pre[pre[:] > 255] = 255
    pre[pre[:] < 0] = 0
    pre = pre.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    img[6: -6, 6: -6, 0] = pre[0, :, :, 0]
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(self.output_path, img)

    # psnr calculation:
    # im1 = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2YCrCb)[6: -6, 6: -6, 0]
    # im2 = cv2.imread(INPUT_NAME, cv2.IMREAD_COLOR)
    # im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2YCrCb)[6: -6, 6: -6, 0]
    # im3 = cv2.imread(output_path, cv2.IMREAD_COLOR)
    # im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2YCrCb)[6: -6, 6: -6, 0]