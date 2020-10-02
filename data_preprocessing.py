'''
Created on Fri Sep 18 10:21:19 2020

@author: Bogdan
'''
import os
import shutil
import numpy as np
import utils

def train_test_split(train_size=0.8, random_state=None):
    ''' Splits the data specified in the top_brands_samples_information
    dictionary into random training and testing subsets

    Arguments:
        *train_size* (float) -- specifies the percentage of the data to be
        selected for the training set; value must be between 0 and 1

        *random_state* (int) -- specifies the seed to be used with the
        RandomState instance, so that the results are reproducible

    Returns:
        tuple of two lists -- the training and testing data
    '''
    # Initialize necessary variables
    np.random.seed(random_state)
    train_image_names, test_image_names = [], []

    # Stratify the data by brand, model and year
    samples_information = utils.read_dictionary('top_10_brands_samples_information.txt')

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

    return train_image_names, test_image_names

def copy_files_helper(training_set_subdirectory, class_name, file_names):
    ''' Helper function which copies the files specified in file_names, from
    the original dataset directory to the specified subset of the training data
    directory and class name directory '''
    for file_name in file_names:
        original_image_path = os.path.join(utils.DATASET_LOCATION, class_name, file_name)
        new_image_path = os.path.join(utils.TRAINING_DIR, training_set_subdirectory, class_name, file_name)

        if os.path.exists(original_image_path + '.jpg'):
            extension = '.jpg'
        else:
            extension = '.png'

        original_image_path += extension
        new_image_path += extension

        shutil.copyfile(original_image_path, new_image_path)

def create_training_data_directory_structure(train_names, test_names, delete_dataset=False):
    ''' Creates the directory structure necessary for loading the training data
    in Keras

    Arguments:
        *train_names* (list of str) -- specifies the names of the files from
        the training set

        *test_names* (list of str) -- specifies the names of the files from
        the test set

        *delete_dataset* (boolean) -- specifies whether the original dataset
        directory should be deleted or not

    Returns:
        Nothing
    '''
    # Create the base directory containing the training and testing sets directories
    try:
        os.mkdir(utils.TRAINING_DIR)
    except FileExistsError:
        # Delete the directory and all its contents and create it anew
        shutil.rmtree(utils.TRAINING_DIR, ignore_errors=True)
        os.mkdir(utils.TRAINING_DIR)

    os.mkdir(os.path.join(utils.TRAINING_DIR, utils.TRAIN_SET_LOCATION))
    os.mkdir(os.path.join(utils.TRAINING_DIR, utils.TEST_SET_LOCATION))

    # For each class, create a corresponding directory in both the train
    # and test directories and copy the specified files in them
    class_names = set([image_name.split('_')[0] for image_name in test_names])

    for class_name in class_names:
        os.mkdir(os.path.join(utils.TRAINING_DIR, utils.TRAIN_SET_LOCATION, class_name))
        os.mkdir(os.path.join(utils.TRAINING_DIR, utils.TEST_SET_LOCATION, class_name))

        print('> Copying train images for {}...'.format(class_name))
        train_files = [train_name for train_name in train_names if train_name.startswith(class_name)]
        copy_files_helper(utils.TRAIN_SET_LOCATION, class_name, train_files)

        print('> Copying test images for {}...'.format(class_name))
        test_files = [test_name for test_name in test_names if test_name.startswith(class_name)]
        copy_files_helper(utils.TEST_SET_LOCATION, class_name, test_files)

    # Delete the original dataset, if required
    if delete_dataset is True:
        shutil.rmtree(utils.DATASET_LOCATION)

def load_data():
    ''' '''
    pass

# Split the data files into training and testing sets
train_image_names, test_image_names = train_test_split(0.8, 64)

# Create directory structure for loading the training data in Keras
create_training_data_directory_structure(train_image_names, test_image_names, True)
