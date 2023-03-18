import sys
sys.path.append("../cycle/stable-diffusion/")
import tensorflow as tf
import matplotlib.pyplot as plt
from textwrap import wrap
from data_helper import *
import keras_cv

import unittest

from unittest.mock import patch

class Model_Test(unittest.TestCase):
    def test_load(self):
        weights_path = tf.keras.utils.get_file(
        origin="https://huggingface.co/sayakpaul/kerascv_sd_pokemon_finetuned/resolve/main/ckpt_epochs_72_res_512_mp_True.h5"
        )

        img_height = img_width = 256
        pokemon_model = keras_cv.models.StableDiffusion(
            img_width=img_width, img_height=img_height
        )
        # We just reload the weights of the fine-tuned diffusion model.
        pokemon_model.diffusion_model.load_weights(weights_path)

        prompts = ["Yoda", "Hello Kitty", "A pokemon with red eyes"]
        images_to_generate = 3
        outputs = []

        for prompt in prompts:
            generated_images = pokemon_model.text_to_image(
                prompt, batch_size=images_to_generate, unconditional_guidance_scale=40
            )
            outputs=outputs+[img for img in generated_images]
        
        for i in range(images_to_generate*len(prompts)):
            plt.subplot(images_to_generate, len(prompts), i+1)
            plt.imshow(outputs[i])

        plt.savefig("sd_model_test.png")
        
if __name__ == '__main__':
    test=Model_Test()
    test.test_load()