import sys
sys.path.append("../cycle/stable-diffusion/")
import tensorflow as tf
import matplotlib.pyplot as plt
from textwrap import wrap
from data_helper import *

import unittest

from unittest.mock import patch

class Data_Loader_Test(unittest.TestCase):
    def training_data_test(self):
        training_dataset= get_data_loader(True)
        plt.figure(figsize=(20, 10))
        sample_batch = next(iter(training_dataset))
        for i in range(3):
            ax = plt.subplot(1, 4, i + 1)
            plt.imshow((sample_batch["images"][i] + 1) / 2)

            text = tokenizer.decode(sample_batch["tokens"][i].numpy().squeeze())
            text = text.replace("<|startoftext|>", "")
            text = text.replace("<|endoftext|>", "")
            text = "\n".join(wrap(text, 12))
            plt.title(text, fontsize=15)

            plt.axis("off")
        plt.savefig("sd_data_loader_test.png")
        print("all done")

if __name__ == '__main__':
    Data_Loader_Test().training_data_test()
