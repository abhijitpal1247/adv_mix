
import numpy as np
import tensorflow as tf


class ApplyBias(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ApplyBias, self).__init__(**kwargs)

    def build(self, input_shape):
        self.len2 = True if len(input_shape) == 2 else False
        b_init = tf.zeros(shape=(input_shape[1],), dtype=tf.dtypes.float32)
        self.b = tf.Variable(b_init, name='b', trainable=True)

    def call(self, inputs, training=None, mask=None):
        b = self.b
        if self.len2:
            x = inputs + b
        else:
            x = inputs + tf.reshape(b, shape=[1, -1, 1, 1])
        return x

    def get_config(self):
        config = super(ApplyBias, self).get_config()
        config.update({
            'len2': self.len2,
        })
        return config
