from sklearn.metrics import confusion_matrix, recall_score
from deep_model_feature_reader import *
import numpy as np
import os
import pickle
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Dropout, concatenate, Input, Lambda, Conv2D, Flatten, Permute, Reshape, multiply, Activation, add, dot, Conv1D, average, maximum, GRU, Embedding
from keras.layers.pooling import MaxPooling1D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianDropout, GaussianNoise
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras import metrics
from keras import backend as K
import numpy as np
from snore_data_extractor import *
import tensorflow as tf
from keras import losses
from sklearn.metrics import recall_score, confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
import sys


def image_entry_model_1(time_steps, data_dim):

    inputs = Input(shape=(time_steps, data_dim, 1))

    x_0 = Reshape((time_steps*data_dim, ))(inputs)

    container = []
    for _ in range(time_steps):
        container.append(Reshape((1, 256))(Dense(256, activation="relu")(x_0)))
    x_0 = concatenate(container, axis=1)
    x_1 = GRU(256, return_sequences=True)(x_0)
    x_2 = GRU(128, return_sequences=True)(x_1)
    x_3 = GRU(64)(x_2)

    prediction = Dense(4, activation="softmax")(x_3)

    model = Model(inputs=inputs, outputs=prediction)

    model.summary()

    '''
    train = snore_data_extractor(load_folder_path, one_hot=True, data_mode="train", resize=(data_dim, time_steps), timechain=False, duplicate=True)
    devel = snore_data_extractor(load_folder_path, one_hot=True, data_mode="devel", resize=(data_dim, time_steps), timechain=False, duplicate=False)
    epoch_num = 500
    batch_size = 16
    loss = tf.reduce_mean(losses.kullback_leibler_divergence(labels, predicts))
    train_step = tf.train.RMSPropOptimizer(learning_rate=0.001, momentum=0.01).minimize(loss)
    '''

    return model


def image_entry_model_2(time_steps, data_dim):

    inputs = Input(shape=(time_steps, data_dim, 1))

    x_0 = Reshape((time_steps*data_dim, ))(inputs)

    container = []
    for _ in range(time_steps):
        container.append(Reshape((1, 256))(Dense(256, activation="relu")(x_0)))
    x_0 = concatenate(container, axis=1)
    x_1 = GRU(256, return_sequences=True)(x_0)
    x_2 = GRU(128)(x_1)

    prediction = Dense(4, activation="softmax")(x_2)

    model = Model(inputs=inputs, outputs=prediction)

    model.summary()

    '''
    train = snore_data_extractor(load_folder_path, one_hot=True, data_mode="train", resize=(data_dim, time_steps), timechain=False, duplicate=True)
    devel = snore_data_extractor(load_folder_path, one_hot=True, data_mode="devel", resize=(data_dim, time_steps), timechain=False, duplicate=False)
    epoch_num = 500
    batch_size = 16
    loss = tf.reduce_mean(losses.kullback_leibler_divergence(labels, predicts))
    train_step = tf.train.RMSPropOptimizer(learning_rate=0.001, momentum=0.01).minimize(loss)
    '''

    return model


def image_entry_model_3(time_steps, data_dim):

    inputs = Input(shape=(time_steps, data_dim, 1))

    x_0 = Reshape((time_steps*data_dim, ))(inputs)

    container = []
    for _ in range(time_steps):
        container.append(Reshape((1, 256))(Dense(256, activation="relu")(x_0)))
    x_0 = concatenate(container, axis=1)
    x_1 = GRU(256)(x_0)

    prediction = Dense(4, activation="softmax")(x_1)

    model = Model(inputs=inputs, outputs=prediction)

    model.summary()

    '''
    train = snore_data_extractor(load_folder_path, one_hot=True, data_mode="train", resize=(data_dim, time_steps), timechain=False, duplicate=True)
    devel = snore_data_extractor(load_folder_path, one_hot=True, data_mode="devel", resize=(data_dim, time_steps), timechain=False, duplicate=False)
    epoch_num = 500
    batch_size = 16
    loss = tf.reduce_mean(losses.kullback_leibler_divergence(labels, predicts))
    train_step = tf.train.RMSPropOptimizer(learning_rate=0.001, momentum=0.01).minimize(loss)
    '''

    return model

if __name__ == "__main__":


    # some basic setups of model and path
    ############################################################################
    model_name = "1"
    num_classes = 4
    time_steps = 16
    data_dim = 256
    load_file_path = "./snore_spectrogram_4_AlexNet_f1/"
    log_path = "./spectro_4_GRU_"+model_name+"_log_dev"


    # extracting the features collected from deep models
    ############################################################################
    train = deep_model_feature_reader(load_file_path, data_mode="train", timechainParameter=False, one_hot=True, duplicate=True)
    x_train, y_train = train.full_data()
    x_train = x_train.reshape((-1, time_steps, data_dim, 1))

    devel = deep_model_feature_reader(load_file_path, data_mode="devel", timechainParameter=False, one_hot=True, duplicate=True)
    x_devel, y_devel = devel.full_data()
    x_devel = x_devel.reshape((-1, time_steps, data_dim, 1))

    test = deep_model_feature_reader(load_file_path, data_mode="test", timechainParameter=False, one_hot=True)
    x_test, y_test = test.full_data()
    x_test = x_test.reshape((-1, time_steps, data_dim, 1))


    # some basic setups of training
    ############################################################################
    datagen = ImageDataGenerator()

    epoch_num = 500
    batch_size = 16

    labels = tf.placeholder(tf.float32, shape=(None, num_classes))
    if model_name == "3":
    	model = image_entry_model_3(time_steps, data_dim)
    elif model_name == "2":
        model = image_entry_model_2(time_steps, data_dim)
    elif model_name == "1":
        model = image_entry_model_1(time_steps, data_dim)
    predicts = model.output
    inputs = model.input
    loss = tf.reduce_mean(losses.kullback_leibler_divergence(labels, predicts))
    UAR_value = tf.constant(0.0)
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("UAR", UAR_value)
    train_step = tf.train.RMSPropOptimizer(learning_rate=0.001, momentum=0.01).minimize(loss)
    init = tf.global_variables_initializer()


    # the progress of training
    ############################################################################
    with tf.Session() as sess:

        sess.run(init)
        K.set_session(sess)
        train_batcher = datagen.flow(x_train, y_train, batch_size=batch_size)
        batch_num = int(np.floor(x_train.shape[0]/batch_size))
        merged = tf.summary.merge_all()
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        tensorboard_train = tf.summary.FileWriter(log_path+"/train")
        tensorboard_devel = tf.summary.FileWriter(log_path+"/devel")
        tensorboard_test = tf.summary.FileWriter(log_path+"/test")
        score = 0.0
        for each_epoch in range(epoch_num):
            print ("############################################################################################")
            print ("This is epoch ", each_epoch, "\n")
            batch_counter = 0
            for x_batch, y_batch in train_batcher:
                if batch_counter == batch_num:
                    break
                sess.run(train_step, feed_dict={labels: y_batch, inputs: x_batch, K.learning_phase(): 1})
                batch_counter += 1
            current_loss = sess.run(loss, feed_dict={labels: y_train, inputs: x_train, K.learning_phase(): 0})
            print ("This is the current loss: ", current_loss)
            print ("\n")
            predict_train = sess.run(predicts, feed_dict={inputs: x_train, K.learning_phase(): 0})
            train_UAR = recall_score(np.argmax(y_train, axis=1), np.argmax(predict_train, axis=1), average="macro")
            print ("This is the current train UAR: ", train_UAR)
            print (confusion_matrix(np.argmax(y_train, axis=1), np.argmax(predict_train, axis=1)))
            print ("\n")
            predict_devel = sess.run(predicts, feed_dict={inputs: x_devel, K.learning_phase(): 0})
            devel_UAR = recall_score(np.argmax(y_devel, axis=1), np.argmax(predict_devel, axis=1), average="macro")
            print ("This is the current devel UAR: ", devel_UAR)
            print (confusion_matrix(np.argmax(y_devel, axis=1), np.argmax(predict_devel, axis=1)))
            print ("\n")

            predict_test = sess.run(predicts, feed_dict={inputs: x_test, K.learning_phase(): 0})
            test_UAR = recall_score(np.argmax(y_test, axis=1), np.argmax(predict_test, axis=1), average="macro")
            print ("This is the current test UAR: ", test_UAR)
            print (confusion_matrix(np.argmax(y_test, axis=1), np.argmax(predict_test, axis=1)))
            print ("\n")

            summary_train = sess.run(merged, feed_dict={labels: y_train, inputs: x_train, UAR_value: train_UAR, K.learning_phase(): 0})
            tensorboard_train.add_summary(summary_train, each_epoch)
            summary_devel = sess.run(merged, feed_dict={labels: y_devel, inputs: x_devel, UAR_value: devel_UAR, K.learning_phase(): 0})
            tensorboard_devel.add_summary(summary_devel, each_epoch)
            summary_test = sess.run(merged, feed_dict={labels: y_test, inputs: x_test, UAR_value: test_UAR, K.learning_phase(): 0})
            tensorboard_test.add_summary(summary_test, each_epoch)


            if devel_UAR > score:
                print ("This model is better than previous ones, so save it!")
                model.save_weights("LSTM_weights_spectro_4_GRU_"+model_name+"_dev.h5")
                score = devel_UAR
