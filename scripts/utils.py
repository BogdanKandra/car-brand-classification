'''
Created on Tue Sep 22 08:54:15 2020

@author: Bogdan

This script contains utility functions
'''
import itertools
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras.applications as apps


### Constants
PROJECT_PATH = os.getcwd()
while os.path.basename(PROJECT_PATH) != 'car-brand-classification':
    PROJECT_PATH = os.path.dirname(PROJECT_PATH)
ORIGINAL_DATASET_LOCATION = os.path.join(os.path.dirname(PROJECT_PATH), 'Data', 'Cars')
DATASET_LOCATION = os.path.join(PROJECT_PATH, 'dataset')
FIGURES_LOCATION = os.path.join(PROJECT_PATH, 'figures')
MODELS_LOCATION = os.path.join(PROJECT_PATH, 'models')
TRAINING_RESULTS_DIR = os.path.join(PROJECT_PATH, 'training_results')
PICKLES_LOCATION = os.path.join(PROJECT_PATH, 'pickles')
TEXTS_LOCATION = os.path.join(PROJECT_PATH, 'texts')
TRAINING_DIR = os.path.join(PROJECT_PATH, 'training_data')
TEST_SET_LOCATION = os.path.join(TRAINING_DIR, 'test')
TRAIN_SET_LOCATION = os.path.join(TRAINING_DIR, 'train')
VALIDATION_SET_LOCATION = os.path.join(TRAINING_DIR, 'validation')
AUGMENTED_DIR = os.path.join(PROJECT_PATH, 'augmented_data')
TEST_AUGMENT_LOCATION = os.path.join(AUGMENTED_DIR, 'test')
TRAIN_AUGMENT_LOCATION = os.path.join(AUGMENTED_DIR, 'train')
VALIDATION_AUGMENT_LOCATION = os.path.join(AUGMENTED_DIR, 'validation')
TEACHABLE_MACHINE_DIR = os.path.join(PROJECT_PATH, 'teachable_machine_data')

ALL_BRANDS_COUNTS_NAME = 'all_brands_samples_counts.txt'
TOP10_BRANDS_COUNTS_NAME = 'top_10_brands_samples_counts.txt'
TOP10_BRANDS_INFORMATION_NAME = 'top_10_brands_samples_information.txt'
SUBSAMPLE_ARRAY_NAME = 'subsample.npy'

BALANCED_DATASET_CREATION = True
TRAIN_SET_PERCENTAGE = 0.7
VALIDATION_SET_PERCENTAGE = 0.1
SUBSAMPLE_PERCENTAGE = 0.1
TEACHABLE_MACHINE_TRAIN_SUBSAMPLE_PERCENTAGE = 0.4
TEACHABLE_MACHINE_TRAIN_SUBSAMPLE_COUNT = 50000
RANDOM_STATE = 64
RESIZE_HEIGHT = 128
RESIZE_WIDTH = 128
NUM_EPOCHS = 50
BATCH_SIZE = 16

MODULE_TO_NETWORKS = {
    'efficientnet': ['EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2'],
    'mobilenet': ['MobileNet'],
    'mobilenet_v2': ['MobileNetV2']
}

### Utility functions
def save_bar_plot(title, xlabel, ylabel, xdata, ydata, color='r', plot_name='figure'):
    ''' Generates a bar plot using the given data and saves it to disk '''
    plt.figure(0, figsize=(19.2, 10.8))
    plt.title(title)
    plt.xlabel(xlabel, fontweight='bold')
    plt.ylabel(ylabel, fontweight='bold')
    plt.bar(xdata, ydata, color=color)
    if os.path.isdir(FIGURES_LOCATION) is False:
        os.mkdir(FIGURES_LOCATION)
    plt.savefig(os.path.join(FIGURES_LOCATION, plot_name + '.png'), dpi=300)
    plt.close()

def show_bar_plot(figure_index, title, xlabel, ylabel, xdata, ydata, color='r'):
    ''' Generates a bar plot using the given data and displays it '''
    plt.figure(figure_index, figsize=(19.2, 10.8))
    plt.title(title)
    plt.xlabel(xlabel, fontweight='bold')
    plt.ylabel(ylabel, fontweight='bold')
    plt.bar(xdata, ydata, color=color)
    plt.show()

def write_dictionary(dictionary, file_name):
    ''' Writes a dictionary to the specified file, in indented JSON format '''
    if os.path.isdir(TEXTS_LOCATION) is False:
        os.mkdir(TEXTS_LOCATION)
    with open(os.path.join(TEXTS_LOCATION, file_name), 'w') as f:
        f.write(json.dumps(dictionary, indent=4))

def read_dictionary(file_name):
    ''' Reads the dictionary specified in JSON format from the specified file '''
    with open(os.path.join(TEXTS_LOCATION, file_name), 'r') as f:
        dictionary = json.load(f)
    
    return dictionary

def save_numpy_array(array, file_name):
    ''' Writes a NumPy array to the specified file '''
    if os.path.isdir(PICKLES_LOCATION) is False:
        os.mkdir(PICKLES_LOCATION)
    with open(os.path.join(PICKLES_LOCATION, file_name), 'wb') as f:
        np.save(f, array, allow_pickle=False, fix_imports=False)

def load_numpy_array(file_name):
    ''' Reads the NumPy array from the specified file '''
    with open(os.path.join(PICKLES_LOCATION, file_name), 'rb') as f:
        array = np.load(f, fix_imports=False)

    return array

def load_pretrained_network(network_name):
    ''' Loads the specified pretrained network from Keras applications, with
    frozen weights '''
    image_shape = (RESIZE_HEIGHT, RESIZE_WIDTH, 3)
    base_model = getattr(apps, network_name)(include_top=False, weights='imagenet', input_shape=image_shape)
    base_model.trainable = False

    return base_model

def load_input_preprocessing_function(module_name):
    ''' Loads the input preprocessing function for the specified pretrained
    network from Keras applications '''
    network_module = getattr(apps, module_name)
    preprocess_input_function = getattr(network_module, 'preprocess_input')

    return preprocess_input_function

def plot_results(training_history, network_name):
    ''' Plots the training and validation accuracy and loss '''
    training_accuracy = training_history['categorical_accuracy']
    validation_accuracy = training_history['val_categorical_accuracy']
    training_loss = training_history['loss']
    validation_loss = training_history['val_loss']

    plt.figure(figsize=(18, 10))
    plt.subplot(2, 1, 1)
    plt.plot(training_accuracy, label='Training Accuracy')
    plt.plot(validation_accuracy, label='Validation Accuracy')
    plt.legend()
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(training_loss, label='Training Loss')
    plt.plot(validation_loss, label='Validation Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Categorical Cross Entropy')
    plt.ylim([0, max(plt.ylim())])
    plt.title('Training and Validation Loss')

    if os.path.isdir(TRAINING_RESULTS_DIR) is False:
        os.mkdir(TRAINING_RESULTS_DIR)

    figure_path = os.path.join(TRAINING_RESULTS_DIR, network_name + ' Training Results.png')
    plt.savefig(figure_path, dpi=300)
    plt.close()

def plot_confusion_matrix(conf_matrix, labels_list, normalized=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    if normalized:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    plt.imshow(conf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels_list))
    plt.xticks(tick_marks, labels_list, rotation=45)
    plt.yticks(tick_marks, labels_list)

    fmt = '.2f' if normalized else 'd'
    thresh = conf_matrix.max() / 2.
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, format(conf_matrix[i, j], fmt), horizontalalignment='center',
                 color='white' if conf_matrix[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig(os.path.join(TRAINING_RESULTS_DIR, title + '.png'), dpi=300)
    plt.close()
