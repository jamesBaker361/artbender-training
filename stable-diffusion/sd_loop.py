from textwrap import wrap
import os

import keras_cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
from keras_cv.models.stable_diffusion.clip_tokenizer import SimpleTokenizer
from keras_cv.models.stable_diffusion.diffusion_model import DiffusionModel
from keras_cv.models.stable_diffusion.image_encoder import ImageEncoder
from keras_cv.models.stable_diffusion.noise_scheduler import NoiseScheduler
from keras_cv.models.stable_diffusion.text_encoder import TextEncoder
from tensorflow import keras

from data_helper import *
from sd_trainer import *
from callbacks import *

import argparse
from datetime import datetime, timezone
import os

parser = argparse.ArgumentParser(description='get some args')
parser.add_argument("--epochs",type=int,help="training epochs", default=2)
parser.add_argument("--test",type=bool, default=False)
parser.add_argument("--batch_size", type=int,default=1)
parser.add_argument("--save_img_parent",type=str,default="/home/jlb638/Desktop/cycle/gen_imgs/stabel_diff/")
parser.add_argument("--name",type=str,default="stable_{}".format(str(datetime.now(timezone.utc))))
parser.add_argument("--save_model_parent", type=str,default="../../../../../scratch/jlb638/artbender-models/stable_diff/")
parser.add_argument("--resolution",type=int,default=512)

args = parser.parse_args()

def objective(trial,args):

    save_folder=args.save_img_parent+args.name
    save_model_folder=args.save_model_parent+args.name
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(save_model_folder, exist_ok=True)

    print(args)

    USE_MP = True
    if USE_MP:
        keras.mixed_precision.set_global_policy("mixed_float16")

    training_dataset= get_data_loader(args.test,args.resolution)

    #image_encoder = ImageEncoder(args.resolution,args.resolution)
    #diffusion_model=DiffusionModel(args.resolution, args.resolution, MAX_PROMPT_LENGTH)


    stable_diffusion_big= keras_cv.models.StableDiffusion(args.resolution, args.resolution)
    image_encoder=stable_diffusion_big.image_encoder
    diffusion_model = stable_diffusion_big.diffusion_model

    if args.resolution == 256:
        vae=tf.keras.Model(
            image_encoder.input,
            image_encoder.layers[-2].output,
        )
    elif args.resolution==512:
        vae=tf.keras.Model(
            image_encoder.input,
            image_encoder.layers[-1].output,
        )
    
    print('vae.output.shape',vae.output.shape)

    diffusion_ft_trainer = Trainer(
        diffusion_model=diffusion_model,
        vae=vae,
        noise_scheduler= NoiseScheduler(),
        use_mixed_precision=USE_MP
    )

    lr = 1e-5
    beta_1, beta_2 = 0.9, 0.999
    weight_decay = (1e-2,)
    epsilon = 1e-08

    optimizer = tf.keras.optimizers.experimental.AdamW(
        learning_rate=lr,
        weight_decay=weight_decay,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
    )
    diffusion_ft_trainer.compile(optimizer=optimizer, loss="mse")

    diffusion_ft_trainer.fit(training_dataset, epochs=args.epochs, callbacks=[
        SaveModelCallback(stable_diffusion_big,save_model_folder),
        GenImgCallback(stable_diffusion_big,save_folder)
    ])

if __name__ == '__main__':
    print("begin!")
    print(args)
    objective(None, args)
    print('end!')