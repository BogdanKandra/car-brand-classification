'''
Created on Tue Sep 22 08:54:15 2020

@author: Bogdan

This script contains utility functions
'''
import json
import os
import matplotlib.pyplot as plt
import numpy as np


### Constants
PROJECT_PATH = os.getcwd()
while os.path.basename(PROJECT_PATH) != 'car-brand-classification':
    PROJECT_PATH = os.path.dirname(PROJECT_PATH)
ORIGINAL_DATASET_LOCATION = os.path.join(os.path.dirname(PROJECT_PATH), 'Data', 'Cars')
DATASET_LOCATION = os.path.join(PROJECT_PATH, 'dataset')
FIGURES_LOCATION = os.path.join(PROJECT_PATH, 'figures')
PICKLES_LOCATION = os.path.join(PROJECT_PATH, 'pickles')
TEXTS_LOCATION = os.path.join(PROJECT_PATH, 'texts')
TRAINING_DIR = os.path.join(PROJECT_PATH, 'training_data')
TEST_SET_LOCATION = os.path.join(TRAINING_DIR, 'test')
TRAIN_SET_LOCATION = os.path.join(TRAINING_DIR, 'train')
VALIDATION_SET_LOCATION = os.path.join(TRAINING_DIR, 'validation')
AUGMENTED_DIR = os.path.join(PROJECT_PATH, 'augmented_data')
TEST_AUGMENT_LOCATION = os.path.join(AUGMENTED_DIR, 'test')
TRAIN_AUGMENT_LOCATION = os.path.join(AUGMENTED_DIR, 'train')

ALL_BRANDS_COUNTS_NAME = 'all_brands_samples_counts.txt'
TOP10_BRANDS_COUNTS_NAME = 'top_10_brands_samples_counts.txt'
TOP10_BRANDS_INFORMATION_NAME = 'top_10_brands_samples_information.txt'
SUBSAMPLE_ARRAY_NAME = 'subsample.npy'

TRAIN_SET_PERCENTAGE = 0.7
VALIDATION_SET_PERCENTAGE = 0.1
SUBSAMPLE_PERCENTAGE = 0.2
RANDOM_STATE = 64
RESIZE_HEIGHT = 128
RESIZE_WIDTH = 128
BATCH_SIZE = 32


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
    plt.savefig(os.path.join(FIGURES_LOCATION, plot_name + '.png'), quality=100)
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
