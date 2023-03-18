import math
import random

import keras_cv
import numpy as np
import tensorflow as tf
from keras_cv import layers as cv_layers
from keras_cv.models.stable_diffusion import NoiseScheduler
from tensorflow import keras
import matplotlib.pyplot as plt
import json

prompts_json=json.load(open('prompts.json','r'))

MAX_PROMPT_LENGTH = 77

def pad_embedding(embedding, sd_tokenizer):
    return embedding + (
        [sd_tokenizer.end_of_text] * (MAX_PROMPT_LENGTH - len(embedding))
    )


def assemble_text_embeddings(prompts, sd_tokenizer, placeholder_token):
    sd_tokenizer.add_tokens(placeholder_token)
    prompts = [prompt.format(placeholder_token) for prompt in prompts]
    embeddings = [sd_tokenizer.encode(prompt) for prompt in prompts]
    embeddings = [np.array(pad_embedding(embedding)) for embedding in embeddings]
    return embeddings
    text_dataset = tf.data.Dataset.from_tensor_slices(embeddings)
    #text_dataset = text_dataset.shuffle(100, reshuffle_each_iteration=True)
    return text_dataset

def assemble_images(hf_dataset, prompts,placeholder_token):
    images=[None for _ in prompts]
    hf_images=[d["image"] for d in hf_dataset if d["name"] == placeholder_token]
    
