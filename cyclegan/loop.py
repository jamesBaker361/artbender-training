import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
from loss_functions import *
from cycle_data_helper import *
from img_helpers import *
import time
import argparse
from datetime import datetime, timezone
import os
from random import randrange
import json

parser = argparse.ArgumentParser(description='get some args')
parser.add_argument("--epochs",type=int,help="training epochs", default=2)
parser.add_argument("--test",type=bool, default=False)
parser.add_argument("--batch_size", type=int,default=1) 
parser.add_argument("--save_img_parent",type=str,default="/home/jlb638/Desktop/cycle/gen_imgs/cyclegan/")
parser.add_argument("--name",type=str,default="cycle_{}".format(str(datetime.now(timezone.utc))))
parser.add_argument("--save_model_parent", type=str,default="../../../../../scratch/jlb638/artbender-models/cyclegan/")
parser.add_argument("--movie_path",type=str,default="jlbaker361/movie_captioned-augmented")
parser.add_argument("--cartoon_path",type=str, default="jlbaker361/avatar-lite_captioned-augmented")

args = parser.parse_args()

def objective(trial,args):
    save_folder=args.save_img_parent+args.name
    save_model_folder=args.save_model_parent+args.name
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(save_model_folder, exist_ok=True)

    print(args)
    OUTPUT_CHANNELS = 3

    generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
    generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

    discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
    discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)

    generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    @tf.function
    def train_step(real_x, real_y):
        # persistent is set to True because the tape is used more than
        # once to calculate the gradients.
        with tf.GradientTape(persistent=True) as tape:
        # Generator G translates X -> Y
        # Generator F translates Y -> X.

            fake_y = generator_g(real_x, training=True)
            cycled_x = generator_f(fake_y, training=True)

            fake_x = generator_f(real_y, training=True)
            cycled_y = generator_g(fake_x, training=True)

            # same_x and same_y are used for identity loss.
            same_x = generator_f(real_x, training=True)
            same_y = generator_g(real_y, training=True)


            disc_real_x = discriminator_x(real_x, training=True)
            disc_real_y = discriminator_y(real_y, training=True)


            disc_fake_x = discriminator_x(fake_x, training=True)
            disc_fake_y = discriminator_y(fake_y, training=True)


            # calculate the loss
            gen_g_loss = generator_loss(disc_fake_y)
            gen_f_loss = generator_loss(disc_fake_x)


            total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)


            # Total generator loss = adversarial loss + cycle loss
            total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
            total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)


            disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)


        # Calculate the gradients for generator and discriminator
        generator_g_gradients = tape.gradient(total_gen_g_loss, 
                                            generator_g.trainable_variables)
        generator_f_gradients = tape.gradient(total_gen_f_loss, 
                                            generator_f.trainable_variables)

        discriminator_x_gradients = tape.gradient(disc_x_loss, 
                                                discriminator_x.trainable_variables)
        discriminator_y_gradients = tape.gradient(disc_y_loss, 
                                                discriminator_y.trainable_variables)

        # Apply the gradients to the optimizer
        generator_g_optimizer.apply_gradients(zip(generator_g_gradients, 
                                                generator_g.trainable_variables))

        generator_f_optimizer.apply_gradients(zip(generator_f_gradients, 
                                                generator_f.trainable_variables))

        discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                    discriminator_x.trainable_variables))

        discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                    discriminator_y.trainable_variables))
    #end train step
    
    train_cartoon, train_movie = get_datasets(batch_size=args.batch_size,test=args.test,movie_path=args.movie_path,cartoon_path=args.cartoon_path)
    print('train_cartoon', train_cartoon)
    print('train_movie', train_cartoon)
    #x = cartoon
    #y = movie
    train_movie_sample=next(iter(train_movie))
    movie_list=[t for t in train_movie][1:]
    train_cartoon_sample=next(iter(train_cartoon))
    cartoon_list=[t for t in train_cartoon][1:]
    print('train_cartoon_sample', train_cartoon_sample.shape)

    print("begin training")
    for epoch in range(args.epochs):
        start = time.time()
        n = 0
        for image_x, image_y in tf.data.Dataset.zip((train_cartoon, train_movie)):
            
            train_step(image_x, image_y)
            if n % 10 == 0:
                print ('.', end='')
            n += 1
        print ('\nTime taken for epoch {} is {} sec\n'.format(epoch + 1,time.time()-start))
        # Using a consistent image so that the progress of the model
        # is clearly visible.
        generate_images(generator_g, train_cartoon_sample,save_folder+"/cartoontomovie{}.png".format(epoch))
        generate_images(generator_f, train_movie_sample,save_folder+"/movietocartoon{}.png".format(epoch))
        random_movie_sample=movie_list[randrange(0,len(movie_list))]
        random_cartoon_sample=cartoon_list[randrange(0,len(cartoon_list))]
        generate_images(generator_g, random_cartoon_sample,save_folder+"/random_cartoontomovie{}.png".format(epoch))
        generate_images(generator_f, random_movie_sample,save_folder+"/random_movietocartoon{}.png".format(epoch))

        if epoch%10==0:
            meta_data = {"epoch":epoch}
            tf.saved_model.save(generator_g,save_model_folder+"generator_g")
            tf.saved_model.save(generator_f,save_model_folder+"generator_f")
            tf.saved_model.save(discriminator_x,save_model_folder+"discriminator_x")
            tf.saved_model.save(discriminator_y,save_model_folder+"discriminator_y")
            json_object = json.dumps(meta_data, indent=4)
 
            # Writing to sample.json
            with open(save_model_folder+"/meta_data.json", "w+") as outfile:
                outfile.write(json_object)

if __name__ == '__main__':
    print("begin!")
    print(args)
    objective(None, args)
    print('end!')