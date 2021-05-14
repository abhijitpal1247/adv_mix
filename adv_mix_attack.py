import pickle
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from imageio import imwrite
from load_models import load_generator
from inference_from_official_weights import main
from catvdogclassifier import classification_model


iterations = int(input("Iterations: "))
#iterations=20
epsilon = float(input("Epsilon(perturbation limit): "))
#epsilon=0.01
latent_dict_file = input("latent file: ")
latent_dict = pickle.load(open(latent_dict_file, 'rb'))

main()
run_item = {
    'res': 256,
    'ckpt_dir': './official-converted/cuda',
    'use_custom_cuda': True,
    'out_fn': None,
}
res = run_item['res']
ckpt_dir = run_item['ckpt_dir']
use_custom_cuda = run_item['use_custom_cuda']
out_fn = run_item['out_fn']
message = f'{res}x{res} with custom cuda' if use_custom_cuda else f'{res}x{res} without custom cuda'
print(message)

resolutions = [4, 8, 16, 32, 64, 128, 256]
feature_maps = [512, 512, 512, 512, 512, 256, 128]
filter_index = resolutions.index(res)
g_params = {
    'z_dim': 512,
    'w_dim': 512,
    'labels_dim': 0,
    'n_mapping': 8,
    'resolutions': resolutions[:filter_index + 1],
    'featuremaps': feature_maps[:filter_index + 1],
}
generator = load_generator(g_params, is_g_clone=True, ckpt_dir=ckpt_dir, custom_cuda=use_custom_cuda)
model = classification_model(256, 256)
model.load_weights('best_weights.hdf5')


alpha = np.zeros((1, 14, 512))
alpha[:, 8:, :] = alpha[:, 8:, :] + 2*(epsilon/iterations)

image_count = 0
bar = tqdm(latent_dict.items())
for file_name, latent in bar:
    gen_latent = tf.identity(latent)
    gen_latent = gen_latent + tf.random.uniform(gen_latent.get_shape().as_list(),
                                                minval=-epsilon, maxval=epsilon,
                                                dtype=tf.dtypes.float32)
    label = tf.zeros(shape=[1, 1])
    orig_latent = latent.numpy()
    for iteration in range(iterations):
        bar.set_description(desc=file_name[:-4] + "  %i/%i" % (iteration + 1, iterations))
        bar.refresh()
        latent_variable = tf.Variable(gen_latent)
        with tf.GradientTape() as tape:
            tape.watch(latent_variable)
            gen_im = generator.synthesis(latent_variable)
            gen_im = tf.transpose(gen_im, [0, 2, 3, 1])
            gen_im = (tf.clip_by_value(gen_im, -1.0, 1.0) + 1.0) * 127.5
            gen_im = tf.clip_by_value(gen_im, 0.0, 255.0)
            prediction = model(gen_im)
            loss = tf.keras.losses.BinaryCrossentropy()(label, prediction)
            grads = tape.gradient(loss, latent_variable)
        signed_grads = tf.sign(grads)
        gen_latent = gen_latent + (alpha * signed_grads)
        gen_latent = tf.clip_by_value(gen_latent, orig_latent - epsilon, orig_latent + epsilon)

    orig_image = generator.synthesis(orig_latent)
    orig_image = (tf.clip_by_value(orig_image, -1.0, 1.0) + 1.0) * 127.5
    orig_image = tf.transpose(orig_image, perm=[0, 2, 3, 1])
    orig_image = tf.cast(orig_image, tf.uint8)
    gen_image = generator.synthesis(gen_latent)
    gen_image = (tf.clip_by_value(gen_image, -1.0, 1.0) + 1.0) * 127.5
    gen_image = tf.transpose(gen_image, perm=[0, 2, 3, 1])
    gen_image = tf.cast(gen_image, tf.uint8)
    image_count += 1
    imwrite('generated/' + file_name[:-4] + '_orig_img.png',
            orig_image[0].numpy())
    imwrite('generated/' + file_name[:-4] + '_gen_img.png',
            gen_image[0].numpy())
