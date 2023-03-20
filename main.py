from wgan import WassersteinGAN
import tensorflow as tf
import numpy as np
import sys
import pydicom as dicom
import os
import matplotlib.pyplot as plt

data_dir = sys.argv[1]

def augment(image):
    p_spatial = tf.random.uniform([], 0, 1.0, dtype = tf.float32)
    p_rotate = tf.random.uniform([], 0, 1.0, dtype = tf.float32)
    p_pixel_1 = tf.random.uniform([], 0, 1.0, dtype = tf.float32)
    p_pixel_2 = tf.random.uniform([], 0, 1.0, dtype = tf.float32)
    p_pixel_3 = tf.random.uniform([], 0, 1.0, dtype = tf.float32)
    
    # Flips
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    
    if p_spatial > .75:
        image = tf.image.transpose(image)
        
    # Rotates
    if p_rotate > .75:
        image = tf.image.rot90(image, k = 3) # rotate 270º
    elif p_rotate > .5:
        image = tf.image.rot90(image, k = 2) # rotate 180º
    elif p_rotate > .25:
        image = tf.image.rot90(image, k = 1) # rotate 90º
        
    # Pixel-level transforms
    if p_pixel_1 >= .4:
        image = tf.image.random_saturation(image, lower = .7, upper = 1.3)
    if p_pixel_2 >= .4:
        image = tf.image.random_contrast(image, lower = .8, upper = 1.2)
    if p_pixel_3 >= .4:
        image = tf.image.random_brightness(image, max_delta = .1)
        
    return image


data = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255,
                                                          samplewise_center = True,
                                                          samplewise_std_normalization = True,
                                                          validation_split = 0.2,
                                                          preprocessing_function = augment)
# set as training data

train_gen  = data.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size = 16,
    color_mode = 'rgb',
    shuffle = True,
    class_mode='categorical',
    subset='training') 

# same directory as training data

valid_gen  = data.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size = 16,
    seed = 1,
    color_mode = 'rgb',
    shuffle = False,
    class_mode='categorical',
    subset='validation')

def visualize(generator):
    images = [generator[0][0][i] for i in range(16)]
    fig, axes = plt.subplots(3, 5, figsize = (10, 10))

    axes = axes.flatten()

    for img, ax in zip(images, axes):
        ax.imshow(img.reshape(224, 224, 3).astype("uint8"))
        ax.axis('off')

    plt.tight_layout()
    plt.show()

visualize(train_gen)


