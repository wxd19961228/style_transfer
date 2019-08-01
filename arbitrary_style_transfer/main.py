# Demo - train the style transfer network & use it to generate an image

from __future__ import print_function

from train import train
from infer import stylize
from utils import list_images


IS_TRAINING = False

# for training
TRAINING_CONTENT_DIR = r'D:\wxd\dataset\train2017'
TRAINING_STYLE_DIR = r'D:\wxd\dataset\wikiart\train'
ENCODER_WEIGHTS_PATH = 'vgg19_light.npz'
LOGGING_PERIOD = 20

STYLE_WEIGHTS = [0.1]
MODEL_SAVE_PATHS = [
    'models/comic_sqrt_0.1/style_weight_0.1.ckpt',
]
#models/sqrt_0.1中的style_weight_0.1.ckpt的参数效果还不错，但是对大头照的效果不是很好，所以
#在其他条件都不变的情况下加入了动漫头像，希望能产生更好的效果。——by0319，效果要20号才能看到
# for inferring (stylize)
INFERRING_CONTENT_DIR = 'images/content'
INFERRING_STYLE_DIR = 'images/style'
OUTPUTS_DIR = 'output'


def main():

    if IS_TRAINING:

        content_imgs_path = list_images(TRAINING_CONTENT_DIR)
        style_imgs_path   = list_images(TRAINING_STYLE_DIR)

        for style_weight, model_save_path in zip(STYLE_WEIGHTS, MODEL_SAVE_PATHS):
            print('\n>>> Begin to train the network with the style weight: %.2f\n' % style_weight)

            train(style_weight, content_imgs_path, style_imgs_path, ENCODER_WEIGHTS_PATH, 
                  model_save_path, logging_period=LOGGING_PERIOD, debug=True)

        print('\n>>> Successfully! Done all training...\n')

    else:

        content_imgs_path = list_images(INFERRING_CONTENT_DIR)
        style_imgs_path   = list_images(INFERRING_STYLE_DIR)

        for style_weight, model_save_path in zip(STYLE_WEIGHTS, MODEL_SAVE_PATHS):
            print('\n>>> Begin to stylize images with style weight: %.2f\n' % style_weight)

            stylize(content_imgs_path, style_imgs_path, OUTPUTS_DIR, 
                    ENCODER_WEIGHTS_PATH, model_save_path, 
                    suffix='-' + str(style_weight))

        print('\n>>> Successfully! Done all stylizing...\n')


if __name__ == '__main__':
    main()

