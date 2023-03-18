import sys
sys.path.append("../cycle/cyclegan")
import tensorflow as tf


import unittest

from unittest.mock import patch

from cycle_data_helper import *
import matplotlib.pyplot as plt

class Cycle_Data_Loader_Test(unittest.TestCase):
    def equal_length_test(self):
        big_length=5
        smaller=[_ for _ in range(1)]
        bigger = [_ for _ in range(big_length)]
        smaller, bigger = equal_length(smaller, bigger)
        self.assertEqual(len(bigger), len(smaller))
        self.assertEqual(big_length, len(smaller))

    def equal_length_switched_test(self):
        big_length=5
        smaller=[_ for _ in range(1)]
        bigger = [_ for _ in range(big_length)]
        smaller, bigger = equal_length(bigger, smaller)
        self.assertEqual(len(bigger), len(smaller))
        self.assertEqual(big_length, len(smaller))

    def test_get_datasets(self):
        train_cartoon, train_movie=get_datasets()
        sample_cartoon = next(iter(train_cartoon))
        sample_movie=next(iter(train_movie))
        plt.subplot(221)
        plt.title('cartoon')
        plt.imshow(sample_cartoon[0] * 0.5 + 0.5)

        plt.subplot(222)
        plt.title('cartoon with random jitter')
        plt.imshow(random_jitter(sample_cartoon[0]) * 0.5 + 0.5)

        plt.subplot(223)
        plt.title('movie')
        plt.imshow(sample_movie[0] * 0.5 + 0.5)

        plt.subplot(224)
        plt.title('movie with random jitter')
        plt.imshow(random_jitter(sample_movie[0]) * 0.5 + 0.5)

        plt.savefig("cyclegan_test_get_datasets.png")



if __name__ =="__main__":
    test=Cycle_Data_Loader_Test()
    test.equal_length_test()
    test.equal_length_switched_test()
    test.test_get_datasets()