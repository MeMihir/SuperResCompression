def SRCNN():
    srcnn = tf.keras.models.Sequential()
    srcnn.add(tf.keras.layers.Conv2D(128, (9, 9),
                    activation='relu', padding='valid', input_shape=(None, None, 1)))
    srcnn.add(tf.keras.layers.Conv2D(64, (3, 3),
                    activation='relu', padding='same'))
    srcnn.add(tf.keras.layers.Conv2D(1, (5, 5),
                    activation='linear', padding='valid'))
    adam = tf.keras.optimizers.Adam(lr=0.0003)
    srcnn.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])

    srcnn.load_weights("3051crop_weight_200.h5")
    return srcnn
