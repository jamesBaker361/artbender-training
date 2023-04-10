import sys
sys.path.append("./")
import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix


import unittest

from unittest.mock import patch

from cyclegan.img_helpers import *
from cyclegan.cycle_data_helper import *
import numpy as np
from datasets import load_dataset

class Img_Helper_Test(unittest.TestCase):
    
    def display_imgs_test(self):
        test_x=tf.keras.utils.load_img('cat.png')
        test_y=tf.keras.utils.load_img('cat2.png')
        def generator_g(img):
            return tf.image.rot90(img)
        

        def generator_f(img):
            return tf.image.rot90(img, k=3)
        
        display_imgs(generator_g,generator_f,test_x,test_y, 'display_imgs_test.png')

    def generate_imgs_test(self):
        dataset,_= get_datasets(1,True)
        img=next(iter(dataset))
        print(img.shape)
        generator_g = pix2pix.unet_generator(3, norm_type='instancenorm')
        generate_images(generator_g, img, "generate_imgs_test.png")

    def generate_imgs_test_batch4(self):
        #dataset,_= get_datasets(2,True)
        #img=next(iter(dataset))
        dataset,_= get_datasets(4,True)
        img=next(iter(dataset))
        print(img.shape)
        generator_g = pix2pix.unet_generator(3, norm_type='instancenorm')
        generate_images(generator_g, img, "generate_imgs_test_batch4.png")


if __name__ == '__main__':
    test=Img_Helper_Test()
    test.display_imgs_test()
    test.generate_imgs_test()
    test.generate_imgs_test_batch4()