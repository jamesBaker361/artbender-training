import tensorflow as tf
import numpy as np
from datasets import load_dataset

BUFFER_SIZE = 10
BATCH_SIZE = 1
IMG_WIDTH = 512
IMG_HEIGHT = 512
AUTOTUNE = tf.data.AUTOTUNE

def random_crop(image):
    cropped_image = tf.image.random_crop(
    image, size=[IMG_HEIGHT, IMG_WIDTH, 3])
    return cropped_image

def normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image

def random_jitter(image):
        # resizing to 286 x 286 x 3
    image = tf.image.resize(image, [540, 540],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # randomly cropping to 256 x 256 x 3
    image = random_crop(image)

    # random mirroring
    image = tf.image.random_flip_left_right(image)

    return image

def preprocess_image_train(image):
    image = random_jitter(image)
    image = normalize(image)
    return image

def equal_length(smaller, bigger):
    if len(bigger) < len(smaller):
        bigger, new_smaller = equal_length(bigger,smaller)
    else:
        new_smaller=[]
        for i in range(len(bigger)):
            new_smaller.append(smaller[i%len(smaller)])
    return new_smaller, bigger

def get_datasets(batch_size=BATCH_SIZE,test=False, cartoon_path="jlbaker361/avatar_captioned-augmented", movie_path="jlbaker361/movie_captioned-augmented"):
    if test:
        data_frame_cartoon= [np.array(img) for img in load_dataset("jlbaker361/little_dataset",split="train")["image"]]
        data_frame_movie=[np.array(img) for img in load_dataset("jlbaker361/little_dataset",split="train")["image"]]
    else:
        data_frame_cartoon= [np.array(img) for img in load_dataset(cartoon_path,split="train")["image"]]
        data_frame_movie=[np.array(img) for img in load_dataset(movie_path,split="train")["image"]]
    data_frame_cartoon, data_frame_movie = equal_length(data_frame_cartoon, data_frame_movie)
    train_cartoon= tf.data.Dataset.from_tensor_slices(data_frame_cartoon).map(
        preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
        BUFFER_SIZE).batch(batch_size)
    train_movie = tf.data.Dataset.from_tensor_slices(data_frame_movie).map(
        preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
        BUFFER_SIZE).batch(batch_size)
    return train_cartoon, train_movie