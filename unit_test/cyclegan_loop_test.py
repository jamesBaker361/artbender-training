import sys
sys.path.append("../cycle/cyclegan")
import tensorflow as tf


import unittest

from unittest.mock import patch
from loop import *

class Loop_Test(unittest.TestCase):
    def save_and_load(self):
        generator_g = pix2pix.unet_generator(3, norm_type='instancenorm')
        img=tf.random.normal((1,512,512,3))
        generator_g(img)
        tf.saved_model.save(generator_g,"/scratch/jlb638/artbender-models/test_generator_g")
        new_generator=tf.saved_model.load("/scratch/jlb638/artbender-models/test_generator_g")
        new_generator(img)

    def objective_test(self):
        args.test=True
        args.epochs=2
        objective(None,args)

if __name__ == '__main__':
    test=Loop_Test()
    test.objective_test()