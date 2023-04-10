import matplotlib.pyplot as plt
import numpy as np

def display_imgs(generator_g, generator_f, test_x,test_y,save_path):
    g_x=generator_g(test_x)
    x_hat=generator_f(g_x)
    f_y=generator_f(test_y)
    y_hat=generator_g(f_y)

    imgs=[test_x,g_x,x_hat,test_y,f_y,y_hat]
    titles=["x","g(x)","f(g(x))","y","f(y)","g(f(x))"]

    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.title(titles[i])
        plt.imshow(imgs[i])

    plt.savefig(save_path)
    plt.close()

def generate_images(model, test_input, save_path):
    prediction = model(test_input)

    plt.figure(figsize=(12, 12))

    rows=min(4, len(prediction))
    fig, axs= plt.subplots(rows, 2)
    if len(prediction)==1:
        axs=np.expand_dims(axs,axis=0)
    for c in range(0,rows):

        display_list = [test_input[c], prediction[c]]
        title = ['Input Image', 'Predicted Image']

        for i in range(2):
            axs[c,i].set_title(title[i]+ str(1+i+c))
            # getting the pixel values between [0, 1] to plot it.
            axs[c,i].imshow(display_list[i] * 0.5 + 0.5)
            axs[c,i].axis('off')
    fig.savefig(save_path)
    plt.close()

    