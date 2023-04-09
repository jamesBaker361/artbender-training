import sys
sys.path.append("../cycle/stable-diffusion/")
import tensorflow as tf
import matplotlib.pyplot as plt
from textwrap import wrap
from data_helper import *
from sd_loop import *
import keras_cv

import unittest

from unittest.mock import patch

class Stable_Diffusion_Loop_Test(unittest.TestCase):
    def _image_encoder_test(self,resolution):
        resolution=resolution
        stable_diffusion_big= keras_cv.models.StableDiffusion(resolution, resolution)
        image_encoder=stable_diffusion_big.image_encoder
        if resolution == 256:
            vae=tf.keras.Model(
            image_encoder.input,
            image_encoder.layers[-2].output,
        )
        elif resolution==512:
            vae=tf.keras.Model(
                image_encoder.input,
                image_encoder.layers[-1].output,
            )
        
        print('vae.output.shape',vae.output.shape)

    def image_encoder_256_test(self):
        self._image_encoder_test(256)

    def image_encoder_512_test(self):
        self._image_encoder_test(512)

    def objective_test(self):
        args.test=True
        args.epochs=1
        args.resolution=256
        objective(None, args)
        print("all done :)")

if __name__ == '__main__':
    test=Stable_Diffusion_Loop_Test()
    test.image_encoder_256_test()
    test.image_encoder_512_test()
    test.objective_test()
    print("all done :)))")