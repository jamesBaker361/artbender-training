import sys
sys.path.append("../cycle/text-inversion")
import tensorflow as tf
import matplotlib.pyplot as plt
from textwrap import wrap
from tex_inv_img_helper import *
from datasets import load_dataset
import keras_cv

import unittest

from unittest.mock import patch

class Img_Helper_Test(unittest.TestCase):
    def __init__(self):
        self.stable_diffusion = keras_cv.models.StableDiffusion()

    def plot_images_test(self):
        generated = self.stable_diffusion.text_to_image("a happy man with a cat", seed=1337, batch_size=3)
        plot_images(generated, 'text_inv_test.png')
        
if __name__ == '__main__':
    test=Img_Helper_Test()
    test.plot_images_test()
    print("all done :)))")