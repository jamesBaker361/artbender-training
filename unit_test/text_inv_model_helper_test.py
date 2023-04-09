import sys
sys.path.append("../cycle/text-inversion")
import tensorflow as tf
import matplotlib.pyplot as plt
from textwrap import wrap
from texinv_model_helper import *
from datasets import load_dataset
import keras_cv

import unittest

from unittest.mock import patch
img_width= img_height = 512

class Model_Helper_Test(unittest.TestCase):
    def add_new_token_test(self):
        stable_diffusion = keras_cv.models.StableDiffusion()
        add_new_token(stable_diffusion, "cat")
        add_new_token(stable_diffusion, "angry cat going for a walk")

if __name__=='__main__':
    test=Model_Helper_Test()
    test.add_new_token_test()
    print("all done :)))")