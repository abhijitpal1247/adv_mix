import tensorflow as tf


def classification_model(img_height, img_width):
    model1 = tf.keras.Sequential()
    model1.add(tf.keras.layers.Input(shape=(img_height, img_width, 3)))
    model1.add(tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255))
    model1.add(tf.keras.layers.Conv2D(16, (3, 3), padding='same'))
    model1.add(tf.keras.layers.BatchNormalization())
    model1.add(tf.keras.layers.Activation('relu'))

    model1.add(tf.keras.layers.Conv2D(16, (3, 3), padding='same'))
    model1.add(tf.keras.layers.BatchNormalization())
    model1.add(tf.keras.layers.Activation('relu'))
    model1.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model1.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same'))
    model1.add(tf.keras.layers.BatchNormalization())
    model1.add(tf.keras.layers.Activation('relu'))

    model1.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same'))
    model1.add(tf.keras.layers.BatchNormalization())
    model1.add(tf.keras.layers.Activation('relu'))
    model1.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model1.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same'))
    model1.add(tf.keras.layers.BatchNormalization())
    model1.add(tf.keras.layers.Activation('relu'))

    model1.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same'))
    model1.add(tf.keras.layers.BatchNormalization())
    model1.add(tf.keras.layers.Activation('relu'))
    model1.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model1.add(tf.keras.layers.Flatten())

    model1.add(tf.keras.layers.Dense(64))
    model1.add(tf.keras.layers.BatchNormalization())
    model1.add(tf.keras.layers.Activation('relu'))
    model1.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model1
