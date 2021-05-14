import tensorflow as tf


def mse_loss(x_gen, x_real):
    return tf.keras.losses.MeanSquaredError()(x_real, x_gen)


class PerceptualModel(tf.keras.Model):
    """Defines the VGG16 model for perceptual loss."""

    def __init__(self, img_size, multi_layers=False):
        """Initializes with image size.
    Args:
      img_size: The image size prepared to feed to VGG16, default=256.
      multi_layers: Whether to use the multiple layers output of VGG16 or not.
    """
        super().__init__()

        vgg = tf.keras.applications.vgg16.VGG16(include_top=False, input_shape=(img_size[0], img_size[1], 3))
        if multi_layers:
            layer_ids = [2, 5, 9, 13, 17]
            layer_outputs = [
                tf.keras.layers.Flatten()(vgg.layers[layer_id].output) for layer_id in layer_ids]
            features = tf.keras.layers.Concatenate(axis=-1)(layer_outputs)
        else:
            layer_ids = [13]  # 13 -> conv4_3
            features = [
                tf.keras.layers.Flatten()(vgg.layers[layer_id].output) for layer_id in layer_ids]

        self._model = tf.keras.Model(inputs=vgg.input, outputs=features)

    def call(self, inputs, mask=None):
        return self._model(tf.keras.applications.vgg16.preprocess_input(inputs))

    def compute_output_shape(self, input_shape):
        return self._model.compute_output_shape(input_shape)


def vgg_loss(x_gen, x_real):
    model = PerceptualModel((256, 256))
    pred_gen = model.predict(x_gen)
    pred_real = model.predict(x_real)
    return tf.keras.losses.MeanSquaredError()(pred_real, pred_gen)
