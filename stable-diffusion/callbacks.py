import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

class GenImgCallback(keras.callbacks.Callback):
    def __init__(self,sd_model, save_folder, prompts=["man holding a sword walking at night", "two birds on a branch", "woman playing with cat"], images_to_generate=2, *args, **kwargs):
        super(GenImgCallback, self).__init__(*args, **kwargs)
        self.sd_model=sd_model
        self.save_folder=save_folder
        self.prompts=prompts
        self.images_to_generate=images_to_generate

    def on_epoch_end(self, epoch, logs=None):
        print("end of epoch {}".format(epoch))
        #prompts = ["man holding a sword walking at night", "two birds on a branch", "woman playing with cat"]
        #images_to_generate = 2
        outputs = []

        for prompt in self.prompts:
            generated_images = self.sd_model.text_to_image(
                prompt, batch_size=self.images_to_generate, unconditional_guidance_scale=40
            )
            outputs=outputs+[img for img in generated_images]
        
        for i in range(self.images_to_generate*len(self.prompts)):
            plt.subplot(self.images_to_generate, len(self.prompts), i+1)
            plt.imshow(outputs[i])

        plt.savefig("{}/epoch{}.png".format(self.save_folder, epoch))

class SaveModelCallback(keras.callbacks.Callback):
    def __init__(self, sd_model,save_model_folder) -> None:
        super(SaveModelCallback, self).__init__()
        self.sd_model=sd_model
        self.save_model_folder= save_model_folder

    def on_epoch_end(self,epoch,logs=None):
        tf.saved_model.save(self.sd_model,self.save_model_folder)
        print("saved at {}".format(self.save_model_folder))