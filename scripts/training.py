'''
Created on Sat Jan  9 13:08:51 2021

@author: Bogdan

This script performs the training of the image classifier
'''
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import sklearn.metrics as sk_metrics
import time
import utils
import tensorflow as tf
import tensorflow.keras.applications as apps
from tensorflow.keras import layers, losses, metrics, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


### Set logging level and define logger
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

### Functions
def build_model_pooling_dropout(module_name, network_name):
    ''' Builds the model, starting from a base model specified by the module
    name and network name '''
    # Define layers
    preprocess_input = utils.load_input_preprocessing_function(module_name)
    base_model = utils.load_pretrained_network(network_name)
    # avg_pooling_layer = layers.GlobalAveragePooling2D(name='avg_pooling_layer')
    flatten_layer = layers.Flatten(name='flatten')
    max_pooling_layer = layers.GlobalMaxPooling2D(name='max_pooling_layer')
    specialisation_layer = layers.Dense(128, activation='relu', name='specialisation_layer')
    dropout_layer = layers.Dropout(0.5, name='dropout_layer')
    classification_layer = layers.Dense(10, activation='softmax', name='classification_layer')

    # Define model structure
    inputs = tf.keras.Input(shape=(utils.RESIZE_HEIGHT, utils.RESIZE_WIDTH, 3))
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = max_pooling_layer(x)
    # x = flatten_layer(x)
    x = specialisation_layer(x)
    x = dropout_layer(x)
    outputs = classification_layer(x)
    model = tf.keras.Model(inputs, outputs)

    return model

def build_model_flatten_dense(module_name, network_name):
    ''' Builds the model, starting from a base model specified by the module
    name and network name '''
    # Define test layers
    preprocess_input = utils.load_input_preprocessing_function(module_name)
    base_model = utils.load_pretrained_network(network_name)
    flatten_layer = layers.Flatten(name='flatten')
    specialisation_layer = layers.Dense(256, activation='relu', name='specialisation_layer')
    dropout_layer = layers.Dropout(0.5, name='dropout_layer')
    classification_layer = layers.Dense(10, activation='softmax', name='classification_layer')

    # Define test model
    inputs = tf.keras.Input(shape=(utils.RESIZE_HEIGHT, utils.RESIZE_WIDTH, 3))
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = flatten_layer(x)
    x = specialisation_layer(x)
    x = dropout_layer(x)
    outputs = classification_layer(x)
    model = tf.keras.Model(inputs, outputs)

    return model



##### Algorithm
if __name__ == '__main__':

    ##### Make preparations for training
    LOGGER.info('>>> Making preparations for training...')
    # X_sample = utils.load_numpy_array(utils.SUBSAMPLE_ARRAY_NAME)
    samples_counts = utils.read_dictionary(utils.TOP10_BRANDS_COUNTS_NAME)
    classes_list = sorted(samples_counts.keys())

    # Create Keras data generators and iterators
    # The augmentation is the same for all data sets, so a single generator is used
    data_generator = ImageDataGenerator(
        # featurewise_center=True,
        # featurewise_std_normalization=True
    )
    # data_generator.fit(X_sample)
    # del X_sample

    train_iterator = data_generator.flow_from_directory(
        directory=utils.TRAIN_SET_LOCATION,
        target_size=(utils.RESIZE_HEIGHT, utils.RESIZE_WIDTH),
        color_mode='rgb',
        classes=classes_list,
        class_mode='categorical',
        batch_size=utils.BATCH_SIZE,
        shuffle=True,
        # seed=utils.RANDOM_STATE,
        interpolation='bilinear'
    )

    validation_iterator = data_generator.flow_from_directory(
        directory=utils.VALIDATION_SET_LOCATION,
        target_size=(utils.RESIZE_HEIGHT, utils.RESIZE_WIDTH),
        color_mode='rgb',
        classes=classes_list,
        class_mode='categorical',
        batch_size=utils.BATCH_SIZE,
        shuffle=False,
        # seed=utils.RANDOM_STATE,
        interpolation='bilinear'
    )

    test_iterator = data_generator.flow_from_directory(
        directory=utils.TEST_SET_LOCATION,
        target_size=(utils.RESIZE_HEIGHT, utils.RESIZE_WIDTH),
        color_mode='rgb',
        classes=classes_list,
        class_mode='categorical',
        batch_size=utils.BATCH_SIZE,
        shuffle=False,
        # seed=utils.RANDOM_STATE,
        interpolation='bilinear'
    )

    # Define train parameters
    train_steps = len(train_iterator)
    validation_steps = len(validation_iterator)
    evaluation_steps = len(test_iterator)
    base_learning_rate = 0.00001 # TODO - scheduler for learning rate
    optimizer = optimizers.Adam(learning_rate=base_learning_rate)
    loss_function = losses.CategoricalCrossentropy()
    train_metrics = [metrics.CategoricalAccuracy(), metrics.AUC(), metrics.Precision(), metrics.Recall()]

    ##### Build and train the model
    for module_name in utils.MODULE_TO_NETWORKS.keys():
        for network_name in utils.MODULE_TO_NETWORKS[module_name]:
            # Skip the current network training if already trained
            if any([network_name in file_name for file_name in os.listdir(utils.TRAINING_RESULTS_DIR)]):
                continue

            LOGGER.info('>>> Training the {} model...'.format(network_name))
            # model = build_model_flatten_dense(module_name, network_name)
            model = build_model_pooling_dropout(module_name, network_name)
            model.summary()

            ##### Compile, train and evaluate the model
            model.compile(optimizer=optimizer,
                          loss=loss_function,
                          metrics=train_metrics)

            training_history = model.fit(train_iterator, epochs=utils.NUM_EPOCHS,
                                         verbose=1, validation_data=validation_iterator,
                                         callbacks=[], steps_per_epoch=train_steps,
                                         validation_steps=validation_steps)
            training_history = training_history.history

            test_results = model.evaluate(test_iterator, steps=evaluation_steps,
                                          return_dict=True)

            ##### Generate results
            LOGGER.info('>>> Running the model on the test set and generating results...')

            # Plot the training and validation accuracy and loss and save the train results
            utils.plot_results(training_history, network_name)
            results_name = '{} Training Results.txt'.format(network_name)
            with open(os.path.join(utils.TRAINING_RESULTS_DIR, results_name), 'w') as f:
                f.write(json.dumps(training_history, indent=4))

            # Save the test results
            results_name = '{} Test Results.txt'.format(network_name)
            with open(os.path.join(utils.TRAINING_RESULTS_DIR, results_name), 'w') as f:
                f.write(json.dumps(test_results, indent=4))

            # Generate the classification report
            predictions = model.predict(test_iterator, steps=evaluation_steps)
            y_pred = np.argmax(predictions, axis=1)
            y_test = test_iterator.classes
            class_labels = list(test_iterator.class_indices.keys())

            report = sk_metrics.classification_report(y_test, y_pred, target_names=class_labels)
            report_name = '{} Classification Report.txt'.format(network_name)
            with open(os.path.join(utils.TRAINING_RESULTS_DIR, report_name), 'w') as f:
                f.write(report)

            # Generate and plot the confusion matrix
            cm = sk_metrics.confusion_matrix(y_test, y_pred, labels=list(range(10)))
            cm_title = '{} Confusion Matrix'.format(network_name)
            utils.plot_confusion_matrix(cm, classes_list, title=cm_title)
