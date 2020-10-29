from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, add
import EED
import EES
import cv2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import prepare_data as pd
import numpy
import math

scale = 2

def psnr(target, ref):
    # assume RGB image
    target_data = numpy.array(target, dtype=float)
    ref_data = numpy.array(ref, dtype=float)

    diff = ref_data - target_data
    diff = diff.flatten('C')

    rmse = math.sqrt(numpy.mean(diff ** 2.))

    return 20 * math.log10(255. / rmse)


def model_EEDS():
    _input = Input(shape=(None, None, 1), name='input')
    _EES = EES.model_EES()(_input)
    _EED = EED.model_EED()(_input)
    _EEDS = add([_EED, _EES])

    model = Model(_input, _EEDS)
    model.compile(optimizer=Adam(lr=0.0003), loss='mse')
    return model


def EEDS_predict():
    IMG_NAME = "./input.jpg"
    INPUT_NAME = "input.jpg"
    OUTPUT_NAME = "EEDS_pre.jpg"

    label = cv2.imread(IMG_NAME)
    shape = label.shape

    img = cv2.resize(label, (int(shape[1] / scale), int(shape[0] / scale)), cv2.INTER_CUBIC)
    cv2.imwrite(INPUT_NAME, img)

    EEDS = model_EEDS()
    EEDS.load_weights("./EEDS_check.h5")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y = numpy.zeros((1, img.shape[0], img.shape[1], 1))
    Y[0, :, :, 0] = img[:, :, 0].astype(float) / 255.
    img = cv2.cvtColor(label, cv2.COLOR_BGR2YCrCb)

    pre = EEDS.predict(Y, batch_size=1) * 255.
    pre[pre[:] > 255] = 255
    pre = numpy.uint8(pre)
    img[:, :, 0] = pre[0, :, :, 0]
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(OUTPUT_NAME, img)

    # psnr calculation:
    im1 = cv2.imread(IMG_NAME, cv2.IMREAD_COLOR)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2YCrCb)
    im2 = cv2.imread(INPUT_NAME, cv2.IMREAD_COLOR)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2YCrCb)
    im2 = cv2.resize(im2, (img.shape[1], img.shape[0]))
    cv2.imwrite("Bicubic.jpg", cv2.cvtColor(im2, cv2.COLOR_YCrCb2BGR))
    im3 = cv2.imread(OUTPUT_NAME, cv2.IMREAD_COLOR)
    im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2YCrCb)

    print("Bicubic:")
    print(cv2.PSNR(im1[:, :, 0], im2[:, :, 0]))
    print("EEDS:")
    print(cv2.PSNR(im1[:, :, 0], im3[:, :, 0]))


if __name__ == "__main__":
    EEDS_predict()