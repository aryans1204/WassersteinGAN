from gettext import npgettext
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

class WassersteinGAN(tf.keras.Model):
    def __init__(self, input_size, latent_dim, critic_epoch, batch_size=128):
        super(WassersteinGAN, self).__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.critic_epoch = critic_epoch
        #define generator model with upsampling layers
        self.generator_model = tf.keras.Sequential()
        self.generator_model.add(tf.keras.layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
        self.generator_model.add(tf.keras.layers.BatchNormalization())
        self.generator_model.add(tf.keras.layers.LeakyReLU())

        self.generator_model.add(tf.keras.layers.Reshape((7, 7, 256)))

        self.generator_model.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same'), use_bias=False)
        self.generator_model.add(tf.keras.layers.BatchNormalization())
        self.generator_model.add(tf.keras.layers.LeakyReLU())

        self.generator_model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same'), use_bias=False)
        self.generator_model.add(tf.keras.layers.batchNormalization())
        self.generator_model.add(tf.keras.layers.LeakyReLU())

        self.generator_model.add(tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding="same"), use_bias=False, activation='tanh')

        #define critic model with downsampling layers
        self.critic_model = tf.keras.Sequential()
        self.critic_model.add(tf.keras.layers.ConvD(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
        self.critic_model.add(tf.keras.layers.LeakyReLU())
        self.critic_model.add(tf.keras.layes.Dropout(0.3))
        self.critic_model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        self.critic_model.add(tf.keras.layers.LeakyReLU())
        self.critic_model.add(tf.keras.layers.Dropout(0.3))
        self.critic_model.add(tf.keras.layers.Flatten())
        self.critic_model.add(tf.keras.layers.Dense(1))

        self._optimizer = tf.keras.Optimizers.RMSProp()
    
    def earthmover(self, real_output, fake_output, mode):
        '''
            Earth mover or Wasserstein 1 distance required to train the WGAN.
        '''
        if mode == "critic":
            return tf.reduce_mean(real_output) - tf.reduce_mean(fake_output)
        
        else:
            return tf.reduce_mean(fake_output)

    
    def _train_critic(self, real_image, batch_size):
        '''
            Trains the Critic of the WGAN for one epoch, based on the real_image, a batch_size.
        '''

        random_noise = tf.random.normal([batch_size, self.latent_dim])
        sigma = tf.random.uniform(shape=[batch_size, 1, 1, 1], min_val=0, max_val=1) #sample a random tensor of shape (batch_szie, 1, 1, 1) between 0 and 1.
        with tf.GradientTape(persistent=True) as critic_tape:
            with tf.GradientTape() as tape:
                fake_image = self.generator_model([random_noise], training=True)  #generate a fake image based on sampled noise from the latent dimension
                fake_temp = sigma * tf.dtypes.cast(real_image, tf.float32) + (1-sigma)*fake_image
                fake_image_preds = self.critic_model([fake_temp], training=True)  #generate the predictions using the critic for the fake image
            
            grads = tape.gradient(fake_image_preds, fake_temp) #gradients for the fake temp considering fake image w.r.t its predictions
            grads_temp = tf.sqrt(tf.reduce_mean(tf.square(grads), axis=[1, 2, 3]))
            clip = tf.reduce_mean(tf.square(grads_temp-1)) #generating the clip amount to clip ghradeitns between [-1, 1]

            fake_preds = self.critic_model(fake_image, training=True)
            real_preds = self.critic_model(real_image, training=True)

            loss = self.earthmover(real_preds, fake_preds, mode="critic") + clip
        
        grad = critic_tape.gradient(loss, self.critic_model.trainable_variables) #calculate gradients for earthmover loss w.r.t critic model
        self._apply_gradients(zip(grad, self.critic_model.trainable_variables))  #perform RMSProp Gradient Descent Step

        return loss

    def _train_generator(self, batch_size, real_image):
        ''' 
            Trains the Generator of the WGAN for one epoch, based on real_image, a batch_size.
        '''

        random_noise = tf.random.normal([batch_size, self.latent_dim])
        with tf.GradientTape() as generator_tape:
            fake_image = self.generator_model([random_noise], training=True)
            fake_preds = self.critic_model(fake_image, training=True)
            loss = self.earthmover(real_image, fake_preds, mode="generator") #calculates Earth Mover for generator

        grads = generator_tape.gradient(loss, self.generator_model.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self.generator_model.trainable_variables))

        return loss

    def _image_utility(model, epoch, test_input, save_path, save=True):
        '''
            Simple utility for plotting images during training of either generator or discriminator.
        '''
        preds = model.predict(test_input)
        
        f = plt.figure(figsize=(24, 10))
        for i in range(preds.shape[0]):
            axs = plt.subplot(6, 8, i+1)
            plt.imshow(preds[i] * 0.5 + 0.5)
            plt.axis('off')
        if save:
            plt.savefig(os.path.join(save_path, 'image_at_epoch_{:04d}.png'.format(epoch)))
        plt.show()


    def train(self, train_data, num_epochs=40, save_path=None):
        ''' 
            Cumulative training step for WGAN, training both generator and critic together.
        '''
        noise = tf.random.normal([18, self.latent_dim])
        critic_epochs = 0
        c_losses, g_losses = [], []
        for epoch in range(num_epochs):
            for step, (image) in enumerate(train_data):
                batch_size = image.shape[0]
                c_loss = self._train_critic(image, tf.constant(batch_size, dtype=tf.int64))
                c_losses.append(c_loss)
                critic_epochs += 1
                if critic_epochs >= self.critic_epoch:
                    g_loss = self._train_generator(image, tf.constant(batch_size, dtype=tf.int64))
                    g_losses.append(g_loss)
                    critic_epochs = 0
                    self._image_utility(self.generator_model, epoch, [noise], save_path)
        
        return c_losses, g_losses
                


                


