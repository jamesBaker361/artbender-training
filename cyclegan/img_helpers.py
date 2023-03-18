import matplotlib.pyplot as plt

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

def generate_images(model, test_input, save_path):
    prediction = model(test_input)

    plt.figure(figsize=(12, 12))

    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.savefig(save_path)

    