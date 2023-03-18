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
    def objective_test(self):
        args.test=True
        args.epochs=1
        args.resolution=256
        objective(None, args)
        print("all done :)")

if __name__ == '__main__':
    test=Stable_Diffusion_Loop_Test()
    test.objective_test()