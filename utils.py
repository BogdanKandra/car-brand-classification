'''
Created on Tue Sep 22 08:54:15 2020

@author: Bogdan

This script contains utility functions
'''
import json
import os
import matplotlib.pyplot as plt

# Constants
ORIGINAL_DATASET_LOCATION = os.path.join(os.path.dirname(os.getcwd()), 'Data', 'Cars')
DATASET_LOCATION = 'dataset'
FIGURES_LOCATION = 'figures'
TEXTS_LOCATION = 'texts'
TRAINING_DIR = 'training_data'
TEST_SET_LOCATION = os.path.join(TRAINING_DIR, 'test')
TRAIN_SET_LOCATION = os.path.join(TRAINING_DIR, 'train')
AUGMENTED_DIR = 'augmented_data'
TEST_AUGMENT_LOCATION = os.path.join(AUGMENTED_DIR, 'test')
TRAIN_AUGMENT_LOCATION = os.path.join(AUGMENTED_DIR, 'train')
TOP10_BRANDS_COUNTS = 'top_10_brands_samples_counts.txt'
TOP10_BRANDS_INFORMATION = 'top_10_brands_samples_information.txt'
RESIZE_HEIGHT = 224
RESIZE_WIDTH = 224

# Utility functions
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
    ''' Writes a dictionary to the specified file, as an indented JSON '''
    if os.path.isdir(TEXTS_LOCATION) is False:
        os.mkdir(TEXTS_LOCATION)
    with open(os.path.join(TEXTS_LOCATION, file_name), 'w') as f:
        f.write(json.dumps(dictionary, indent=4))

def read_dictionary(file_name):
    ''' Reads the dictionary specified in JSON format from the specified file '''
    with open(os.path.join(TEXTS_LOCATION, file_name), 'r') as f:
        dictionary = json.load(f)
    
    return dictionary
