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


def copy_weights_DtoM(model, discriminator):
    for iterm in discriminator.layers:
        if type(iterm) == keras.engine.topology.InputLayer:
            continue
        else:
            model.get_layer(name=iterm.name).set_weights(iterm.get_weights())
            model.get_layer(name=iterm.name).trainable = False

def copy_weights_GtoM(model, generator):
    for iterm in generator.layers:
        if type(iterm) == keras.engine.topology.InputLayer:
            continue
        else:
            model.get_layer(name=iterm.name).set_weights(iterm.get_weights())

def copy_weights_MtoG(model, generator):
    for iterm in generator.layers:
        if type(iterm) == keras.engine.topology.InputLayer:
            continue
        else:
            iterm.set_weights(model.get_layer(name=iterm.name).get_weights())


            ######################################### begin here ###############################################
            ######################################### begin here ###############################################
            ######################################### begin here ###############################################
            ######################################### begin here ###############################################
            ######################################### begin here ###############################################
            ######################################### begin here ###############################################

# input image dimensions
img_rows, img_cols = 448, 640
input_shape = (img_rows, img_cols, 3)
depth_shape = (img_rows, img_cols, 1)

# initialize the models
model = GAN_models_init.model_init(input_shape)
generator = GAN_models_init.model_g(input_shape)
discriminator = GAN_models_init.model_d()

# compile the models
model.compile(loss=['mean_absolute_error', 'mean_absolute_error', 'mean_absolute_error', 'mean_absolute_error', 'mean_absolute_error', 'mean_absolute_error',
                    'categorical_crossentropy'],
              metrics=[keras.losses.mean_absolute_error,
                       keras.losses.mean_absolute_error,
                       keras.losses.mean_absolute_error,
                       keras.losses.mean_absolute_error,
                       keras.losses.mean_absolute_error,
                       keras.losses.mean_absolute_error,
                       keras.metrics.categorical_accuracy],
              optimizer=keras.optimizers.Adadelta())
generator.compile(loss='mean_absolute_error',
                  metrics=[utility.metric_L1_real],
                  optimizer=keras.optimizers.Adadelta())
discriminator.compile(loss='categorical_crossentropy',
                      metrics=[keras.metrics.categorical_accuracy],
                      optimizer=keras.optimizers.Adadelta())

# visualize the model
plot_model(model, to_file='model.png')

########################################### main loop ##################################################################
########################################### main loop ##################################################################
########################################### main loop ##################################################################
########################################### main loop ##################################################################
########################################### main loop ##################################################################
########################################### main loop ##################################################################
########################################### main loop ##################################################################
n = 0
train_discriminator_steps = 10
record_g_train = []
record_g_val = []
record_d_test1 = []
record_d_test2 = []
while True:
    ######################################### begin here ###############################################
    ######################################### begin here ###############################################
    ######################################### begin here ###############################################
    ######################################### begin here ###############################################
    ######################################### begin here ###############################################
    ######################################### begin here ###############################################
    if n == 0:
        generator.load_weights('./trained_models/model_epoch_15.hdf5')
        #load the genetator part
        copy_weights_GtoM(model, generator)

    ####################### Train the discriminators (use the generator to generate fake examples)
    ####################### Train the discriminators (use the generator to generate fake examples)
    ####################### Train the discriminators (use the generator to generate fake examples)
    ####################### Train the discriminators (use the generator to generate fake examples)
    ####################### Train the discriminators (use the generator to generate fake examples)
    discriminator = utility.train_GAN_epoch(discriminator, generator, isTrain=True, batchSize = 20, steps=int(train_discriminator_steps))
    train_discriminator_steps += train_discriminator_steps * 0.1
    history = utility.train_GAN_epoch(discriminator, generator, isTrain=False, batchSize = 20, steps=12)
    record_d_test1.append(history)
    # load the discriminator to model
    copy_weights_DtoM(model, discriminator)

    ####################### Train the model #######################
    ####################### Train the model #######################
    ####################### Train the model #######################
    ####################### Train the model #######################
    ####################### Train the model #######################
    ####################### Train the model #######################
    ####################### Train the model #######################
    history = model.fit_generator(utility.data_generator(isTrain = True, isGAN=True, batchSize = 20), steps_per_epoch = 10, epochs = 1)
    record_g_train.append(history.history)
    #load the generator out to generator
    copy_weights_MtoG(model, generator)

    ############################## save informations ###################################
    ############################## save informations ###################################
    ############################## save informations ###################################
    ############################## save informations ###################################
    ############################## save informations ###################################
    ############################## save informations ###################################
    history = generator.evaluate_generator(utility.data_generator(isTrain = False, isGAN = False, batchSize = 20), steps = 25)
    record_g_val.append(history)
    history = utility.train_GAN_epoch(discriminator, generator, isTrain=False, batchSize = 20, steps=12)
    record_d_test2.append(history)
    dict_to_save = {'record_g_val': record_g_val, 'record_g_train': record_g_train, 'record_d_test1': record_d_test1, 'record_d_test2': record_d_test2}
    sio.savemat('record.mat', dict_to_save)

    n += 1

print(' ')

