import numpy as np
import tensorflow as tf

class VGG_LOSS(object):

    def __init__(self, image_shape):
        
        self.image_shape = image_shape

    # computes VGG loss or content loss
    def vgg_loss(self, y_true, y_pred):
    
        vgg19 = tf. VGG19(include_top=False, weights='imagenet', input_shape=self.image_shape)
        vgg19.trainable = False
        # Make trainable as False
        for l in vgg19.layers:
            l.trainable = False
        model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
        model.trainable = False
    
        return K.mean(K.square(model(y_true) - model(y_pred)))

def SRGAN():
    tf.keras.losses.vgg_loss = VGG_LOSS.vgg_loss
    srgan = tf.keras.models.load_model('./gen_model3000.h5', custom_objects={'vgg_loss': VGG_LOSS.vgg_loss})
    return srgan

