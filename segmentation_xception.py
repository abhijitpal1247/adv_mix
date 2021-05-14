import pixellib
import os
from tqdm import tqdm
from imageio import imwrite
import numpy as np
IMAGE_DIR = input("Image Directory: ")
from pixellib.instance import instance_segmentation
segment_image = instance_segmentation()
segment_image.load_model("mask_rcnn_coco.h5")
target_classes = segment_image.select_target_classes(cat=True)
for  file_name in tqdm(os.listdir(IMAGE_DIR), desc="running detections"):
  file_path = os.path.join(IMAGE_DIR, file_name)
  segmask, output = segment_image.segmentImage(file_path, segment_target_classes= target_classes)
  for i in range(len(segmask['class_ids'])):
    if segmask['class_ids'][i] == 16:
      imwrite("generated_masks/mask_" + file_name, (segmask['masks'][:,:,i]*255.0).astype(np.uint8))
    
 
