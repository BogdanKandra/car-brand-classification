'''
Created on Fri Sep 18 10:21:19 2020

@author: Bogdan
'''
import os
import numpy as np
import utils

def train_test_split(train_size=0.8, random_state=None):
    ''' Splits the data specified in the top_brands_samples_information
    dictionary into random training and testing subsets

    Arguments:
        *train_size* (float) -- specifies the percentage of the data to be
        selected for the training set; value must be between 0 and 1

        *random_state* (int) -- specifies the seed to be used with the
        RandomState instance

    Returns:
        tuple of four lists -- the training and testing data and labels
    '''
    # Initialize necessary variables
    np.random.seed(random_state)
    train_image_names, test_image_names, train_labels, test_labels = [], [], [], []

    # Stratify the data by brand, model and year
    samples_information = utils.read_dictionary('top_brands_samples_information.txt')

    for key in samples_information.keys():
        brand, model, year = key.split('|')
        count = samples_information[key]
        training_count = int(train_size * count) if count > 1 else 1

        # Generate indices permutation for selecting train and test data
        available_indices = np.random.permutation(count)
        train_indices = available_indices[:training_count]
        test_indices = available_indices[training_count:] if count > 1 else []

        # Generate and append the relevant image names
        image_base_name = brand + '_' + model + '_' + year + '_'
        key_train_image_names = [image_base_name + str(index) for index in train_indices]
        key_test_image_names = [image_base_name + str(index) for index in test_indices] if count > 1 else []
        train_image_names.extend(key_train_image_names)
        test_image_names.extend(key_test_image_names)

        # Generate and append the labels
        key_train_labels = training_count * [brand]
        key_test_labels = (count - training_count) * [brand]
        train_labels.extend(key_train_labels)
        test_labels.extend(key_test_labels)

    return train_image_names, test_image_names, train_labels, test_labels

X_train, X_test, y_train, y_test = train_test_split(0.8, 64)
