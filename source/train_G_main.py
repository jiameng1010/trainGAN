from __future__ import print_function
import keras
import tensorflow as tf
from keras import backend as K
import numpy as np
import GAN_models_init
import utility
import scipy.io as sio
from keras.utils import plot_model
import matplotlib.pyplot as plt
import cv2


# input image dimensions
img_rows, img_cols = 448, 640
input_shape = (img_rows, img_cols, 3)
depth_shape = (img_rows, img_cols, 1)

# initialize the models
model = GAN_models_init.model_g(input_shape)

# compile the models
model.compile(loss='mean_absolute_error',
              metrics=[keras.losses.mean_absolute_error],
              optimizer=keras.optimizers.Adadelta())

# training
history = []
index_org = sio.loadmat('Data_seperation2.mat')  # load the list of data name
for i in range(1, 40):

    history[i] = model.fit_generator(utility.data_generator(index_org['train_list'], isTrain = True, isGAN = False, batchSize = 10),
                                     steps_per_epoch = 6000,
                                     epochs = 1,
                                     validation_data=utility.data_generator(index_org['val_list'], isTrain=True, isGAN=False, batchSize=20),
                                     validation_steps = 600)
    filename = '../exp_data/trained_models/model_epoch_' + str(i) + '.hdf5'
    model.save_weights(filename)

print('\n')