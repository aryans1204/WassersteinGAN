# WassersteinGAN
This is my implementation of the WassersteinGAN using Tensorflow 2.9

Instructions to use:

`git clone ...` into your working directory

Then simply in your Python file, make an impot statement as such:

`from wgan import WassersteinGAN`

There are multiple utilities provided, but three may be of most importance:

`WassersteinGAN()` model initializer.

`train(train_data, num_epochs)` to train the model on train_data for num_epochs

`predict(input_data)` to perform inference on the trained model.

Experiments have been conducted with the MNIST dataset provided in the Jupyter notebook.
