import numpy as np
import tensorflow as tf


class Downscale2D(tf.keras.layers.Layer):
    def __init__(self, factor=2, **kwargs):
        super(Downscale2D, self).__init__(**kwargs)
        self.factor = factor

    def call(self, inputs, training=None, mask=None):
        if self.factor == 1:
            return inputs
        ksize = [1, 1, self.factor, self.factor]
        return tf.nn.avg_pool(inputs, ksize=ksize, strides=ksize, padding='VALID', data_format='NCHW')

    def get_config(self):
        config = super(Downscale2D, self).get_config()
        return config
