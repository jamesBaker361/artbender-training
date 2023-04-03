import sys
sys.path.append("../cycle/text-inversion")
import tensorflow as tf
import matplotlib.pyplot as plt
from textwrap import wrap
from texinv_data_helper import *
from datasets import load_dataset

import unittest

from unittest.mock import patch

class Data_Helper_Test(unittest.TestCase):
    def assemble_text_embeddings_test(self):
        stable_diffusion = keras_cv.models.StableDiffusion()
        sd_tokenizer=stable_diffusion.tokenizer
        prompts=prompts_json["default_prompts"]
        embeddings=assemble_text_embeddings(prompts, sd_tokenizer, "token_token_token")

    def assemble_images_test(self):
        prompts=prompts_json["default_prompts"]
        hf_dataset=load_dataset("jlbaker361/little_dataset",split="train")
        assemble_images(hf_dataset,prompts,"cat1")

    def assemble_zipped_dataset_test(self):
        prompts=prompts_json["default_prompts"]
        hf_dataset=load_dataset("jlbaker361/little_dataset",split="train")
        stable_diffusion = keras_cv.models.StableDiffusion()
        sd_tokenizer=stable_diffusion.tokenizer
        zipped_dataset =assemble_zipped_dataset(prompts, hf_dataset, sd_tokenizer, ["cat1", "cat2"])
        (img,embedding)=next(iter(zipped_dataset))
        print(img.shape, embedding.shape)

    def get_data_loader_test(self):
        prompt_list_name= "test_prompts"
        stable_diffusion = keras_cv.models.StableDiffusion()
        batch_size =2
        dataset_name = "jlbaker361/little_dataset"
        placeholders = ["cat1", "cat2"]
        data_loader=get_data_loader(prompt_list_name,stable_diffusion,batch_size, dataset_name, placeholders)
        (img,embedding)=next(iter(data_loader))
        print(img.shape, embedding.shape)


if __name__ =='__main__':
    test=Data_Helper_Test()
    test.assemble_text_embeddings_test()
    test.assemble_images_test()
    test.assemble_zipped_dataset_test()
    test.get_data_loader_test()