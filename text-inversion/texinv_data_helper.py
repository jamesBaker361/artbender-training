import math
import random

import keras_cv
import numpy as np
import tensorflow as tf
from keras_cv import layers as cv_layers
from keras_cv.models.stable_diffusion import NoiseScheduler
from tensorflow import keras
import matplotlib.pyplot as plt
from datasets import load_dataset
import json

prompts_json=json.load(open('text-inversion/prompts.json','r'))

MAX_PROMPT_LENGTH = 77

def pad_embedding(embedding, sd_tokenizer):
    return embedding + (
        [sd_tokenizer.end_of_text] * (MAX_PROMPT_LENGTH - len(embedding))
    )


def assemble_text_embeddings(prompts, sd_tokenizer, placeholder_token):
    sd_tokenizer.add_tokens(placeholder_token)
    prompts = [prompt.format(placeholder_token) for prompt in prompts]
    embeddings = [sd_tokenizer.encode(prompt) for prompt in prompts]
    embeddings = [np.array(pad_embedding(embedding, sd_tokenizer)) for embedding in embeddings]
    return embeddings
    text_dataset = tf.data.Dataset.from_tensor_slices(embeddings)
    #text_dataset = text_dataset.shuffle(100, reshuffle_each_iteration=True)
    return text_dataset

def assemble_images(hf_dataset, prompts,placeholder_token):
    images=[None for _ in prompts]
    hf_images=[np.array(d["image"])/127.5 - 1 for d in hf_dataset if d["name"] == placeholder_token]
    for x in range(len(images)):
        images[x]=hf_images[x%len(hf_images)]
    return images
    
def assemble_zipped_dataset(prompts,hf_dataset,sd_tokenizer,placeholders):
    all_embeddings=[]
    all_images=[]
    for placeholder_token in placeholders:
        embeddings=assemble_text_embeddings(prompts, sd_tokenizer, placeholder_token)
        images=assemble_images(hf_dataset, prompts, placeholder_token)
        all_embeddings+=embeddings
        all_images+=images
    image_dataset=tf.data.Dataset.from_tensor_slices(all_images)
    image_dataset = image_dataset.map(
        cv_layers.RandomCropAndResize(
            target_size=(512, 512),
            crop_area_factor=(0.8, 1.0),
            aspect_ratio_factor=(1.0, 1.0),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    image_dataset = image_dataset.map(
        cv_layers.RandomFlip(mode="horizontal"),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    embedding_dataset=tf.data.Dataset.from_tensor_slices(all_embeddings)
    return tf.data.Dataset.zip((image_dataset, embedding_dataset))

def get_data_loader(prompt_list_name,stable_diffusion,batch_size, dataset_name, placeholders):
    hf_dataset=load_dataset(dataset_name, split="train")
    prompts=prompts_json[prompt_list_name]
    sd_tokenizer=stable_diffusion.tokenizer
    for placeholder_token in placeholders:
        sd_tokenizer.add_tokens(placeholder_token)
    zipped_dataset=assemble_zipped_dataset(prompts, hf_dataset, sd_tokenizer, placeholders)
    return zipped_dataset.shuffle(100,1234).batch(batch_size)