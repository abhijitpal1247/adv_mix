# adv_mix
# *Under progress*

Implemenation of "Achieving Robustness in the Wild via Adversarial Mixing With Disentangled Representations"

I have used the StyleGAN implementation by moono (https://github.com/moono/stylegan2-tf-2.x)

And I have used the YOLO V3 implementation by zzh8829 (https://github.com/zzh8829/yolov3-tf2)

Each of data_0 to data_20 contains around 200 images of cat from LSUN dataset and catvdog dataset of a kaggle competition.

Each of these folders contain a pickle file which contains the embedding of the images of that folder in the latent space of StyleGAN.

project.py: contains code regarding the projection of image to StyleGAN latent space.

Image2style_gen folder contains the projected images.

catvdogclassifier.py: classifier trained on the kaggle's catvdog dataset.

generated_classifier folder contains the adversarial images corresponding to the above mentioned classification model.

generated_OD contains the adversarial images corresponding to YOLO V3 object detection model.

adv_mix_attack.py: contains code for adversarial attack against the classification model.

adv_mix_attack_yolo.py: contains code for adversarial attack against the YOLO object detection model.

In generated_OD there are folders corresponding to each PGD setting.

