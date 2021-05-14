import numpy as np
import cv2
import os
from tqdm import tqdm
import PIL
from PIL import Image, ImageOps
folder = input("folder: ")
images = []
for file in tqdm(os.listdir(folder), desc='creating dataset'):
    if os.path.isfile(os.path.join('generated_masks', 'mask_' + file)):
        if 'gen' in file:
            image = Image.open(os.path.join(folder, file))
            image = image.resize((144, 144), PIL.Image.NEAREST)
            image = np.array(image)
            mask = Image.open(os.path.join('generated_masks', 'mask_' + file))
            mask = ImageOps.grayscale(mask)
            mask = mask.resize((144, 144), PIL.Image.NEAREST)
            mask = np.array(mask)
            mask = np.expand_dims(mask, axis=2)
            image_mask = np.concatenate((image, mask), axis=2)
            images.append(image_mask)
images = np.array(images)
np.save('cats_with_masks.npy', images)

