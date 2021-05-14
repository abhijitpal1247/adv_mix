import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from imageio import imwrite
from tqdm import tqdm

folder = input('folder: ')
bg_im_path = input('BG Image: ')
req_x = int(input('Required centre\'s x coordinate: '))
req_y = int(input('Required centre\'s y coordinate: '))
bg_im = Image.open(bg_im_path)
bg_im = np.array(bg_im)
for file in tqdm(os.listdir(folder), desc='creating composites'):
    if os.path.isfile(os.path.join('generated_masks', 'mask_' + file)):
        if 'gen' in file:
            mask = Image.open(os.path.join('generated_masks', 'mask_' + file))
            mask = np.array(mask)
            mask = np.expand_dims(mask, -1)
            mask = tf.image.resize_with_crop_or_pad(mask, bg_im.shape[0], bg_im.shape[1])
            x_min = 1000
            x_max = -1
            y_min = 1000
            y_max = -1
            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    if mask[i, j] == 255:
                        if i < x_min:
                            x_min = i
                        if i > x_max:
                            x_max = i
                        if j < y_min:
                            y_min = j
                        if j > y_max:
                            y_max = j

            centre_x = (x_min + x_max) // 2
            centre_y = (y_min + y_max) // 2
            delta_x = req_x - centre_x
            delta_y = req_y - centre_y
            T = np.float32([[1, 0, delta_y], [0, 1, delta_x]])
            translated_mask = cv2.warpAffine(np.float32(mask), T, (mask.shape[1], mask.shape[0]))
            fg = Image.open(os.path.join(folder, file))
            fg = np.array(fg)
            fg = tf.image.resize_with_crop_or_pad(fg, bg_im.shape[0], bg_im.shape[1])
            fg = fg.numpy()
            translated_fg = cv2.warpAffine(np.float32(fg), T, (fg.shape[1], fg.shape[0]))
            composite_image = (bg_im * (1 - (np.tile(np.expand_dims(translated_mask, -1), 3) // 255))).astype(
                np.uint8) + ((np.tile(np.expand_dims(translated_mask, -1), 3) // 255) * translated_fg).astype(np.uint8)

            x_min = 1000
            x_max = -1
            y_min = 1000
            y_max = -1
            for i in range(translated_mask.shape[0]):
                for j in range(translated_mask.shape[1]):
                    if translated_mask[i, j] == 255:
                        if i < x_min:
                            x_min = i
                        if i > x_max:
                            x_max = i
                        if j < y_min:
                            y_min = j
                        if j > y_max:
                            y_max = j
            with open('ground_truth/' + bg_im_path[:-4] + '.txt') as f:
                with open('ground_truth/' + bg_im_path[:-4] + '_' + file[:-4] + '.txt', "w") as f1:
                    for line in f:
                        f1.write(line)

            with open('ground_truth/' + bg_im_path[:-4] + '_' + file[:-4] + '.txt', 'a+') as file_object:
                file_object.write('cat' + " %d %d %d %d\n" % (x_min, y_min, x_max, y_max))
            imwrite('composites/' + bg_im_path[:-4] + '_' + file, composite_image)

