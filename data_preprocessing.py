'''
Created on Fri Sep 18 10:21:19 2020

@author: Bogdan
'''
import os
import shutil
import numpy as np
import utils
import time
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Constants
HEIGHT = 224
WIDTH = 224
TOP10_BRANDS_COUNTS = 'top_10_brands_samples_counts.txt'
TOP10_BRANDS_INFORMATION = 'top_10_brands_samples_information.txt'

def train_test_split(train_size=0.8, random_state=None):
    ''' Splits the data specified in the top_brands_samples_information
    dictionary into random training and testing subsets

    Arguments:
        *train_size* (float) -- specifies the percentage of the data to be
        selected for the training set; value must be between 0 and 1

        *random_state* (int) -- specifies the seed to be used with the
        RandomState instance, so that the results are reproducible

    Returns:
        tuple of four lists -- the training and testing data and labels
    '''
    # Initialize necessary variables
    np.random.seed(random_state)
    train_image_names, test_image_names, train_labels, test_labels = [], [], [], []

    # Stratify the data by brand, model and year
    samples_information = utils.read_dictionary(TOP10_BRANDS_INFORMATION)

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

def copy_files_helper(training_set_subdirectory, class_name, file_names):
    ''' Helper function which copies the files specified in file_names, from
    the original dataset directory to the specified subset of the training data
    directory and class name directory '''
    for file_name in file_names:
        original_image_path = os.path.join(utils.DATASET_LOCATION, class_name, file_name)
        new_image_path = os.path.join(training_set_subdirectory, class_name, file_name)

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

    os.mkdir(utils.TRAIN_SET_LOCATION)
    os.mkdir(utils.TEST_SET_LOCATION)

    # For each class, create a corresponding directory in both the train
    # and test directories and copy the specified files in them
    class_names = set([image_name.split('_')[0] for image_name in test_names])

    for class_name in class_names:
        os.mkdir(os.path.join(utils.TRAIN_SET_LOCATION, class_name))
        os.mkdir(os.path.join(utils.TEST_SET_LOCATION, class_name))

        print('> Copying train images for {}...'.format(class_name))
        train_files = [train_name for train_name in train_names if train_name.startswith(class_name)]
        copy_files_helper(utils.TRAIN_SET_LOCATION, class_name, train_files)

        print('> Copying test images for {}...'.format(class_name))
        test_files = [test_name for test_name in test_names if test_name.startswith(class_name)]
        copy_files_helper(utils.TEST_SET_LOCATION, class_name, test_files)

    # Delete the original dataset, if required
    if delete_dataset is True:
        shutil.rmtree(utils.DATASET_LOCATION)

def load_data_helper(image_names):
    ''' Helper function which loads the files specified in image_names, from
    the original dataset directory '''
    images = []

    for image_name in image_names:
        class_name = image_name[:image_name.index('_')]
        image_path = os.path.join(utils.DATASET_LOCATION, class_name, image_name)
        if os.path.exists(image_path + '.jpg'):
            extension = '.jpg'
        else:
            extension = '.png'
        image_path += extension

        image = Image.open(image_path).convert('RGB').resize((WIDTH, HEIGHT), Image.BILINEAR)
        images.append(np.array(image))

    return np.array(images)

def load_data(train_image_names, test_image_names):
    ''' Loads the images specified by the *train_image_names* and
    *test_image_names* lists into train and test NumPy arrays

    Arguments:
        *train_image_names* (list of str) -- specifies the names of the files
        to be loaded as the train set

        *test_image_names* (list of str) -- specifies the names of the files
        to be loaded as the test set

    Returns:
        tuple of two NumPy arrays -- the training and testing sets
    '''
    print('> Loading train data...')
    X_train = load_data_helper(train_image_names)
    print('> Loading test data...')
    X_test = load_data_helper(test_image_names)

    return X_train, X_test

if __name__ == '__main__':
    # Split the data files into training and testing sets
    print('>>> Splitting the data into training and testing sets...')
    start_split = time.time()
    train_image_names, test_image_names, y_train, y_test = train_test_split(0.8, 64)
    end_split = time.time()
    print('>>> Splitting took {}'.format(end_split - start_split))





    ########## To delete
    # # Load the data
    # print('>>> Loading the data...')
    # start_load = time.time()
    # X_train, X_test = load_data(train_image_names, test_image_names)
    # end_load = time.time()
    # print('>>> Loading took {}'.format(end_load - start_load))

    # # Create directory structure for loading the training data in Keras
    # create_training_data_directory_structure(train_image_names, test_image_names, True)

    # # Create Keras data generators and iterators
    # samples_counts = utils.read_dictionary(TOP10_BRANDS_COUNTS)

    # data_generator = ImageDataGenerator(
    #     featurewise_center=True,
    #     featurewise_std_normalization=True,
    #     rescale=1./255
    # ) # The augmentation is the same for both train and test sets, so a single generator is used

    # train_generator = data_generator.flow_from_directory(
    #     directory='data/train',
    #     target_size=(224, 224), # Size of MobileNet inputs
    #     color_mode='rgb',
    #     classes=list(samples_counts.keys()),
    #     class_mode='categorical',
    #     batch_size=32,
    #     shuffle=False
    # )
    # test_generator = data_generator.flow_from_directory(
    #     directory='data/test',
    #     target_size=(224, 224), # Size of MobileNet inputs
    #     color_mode='rgb',
    #     classes=list(samples_counts.keys()),
    #     class_mode='categorical',
    #     batch_size=32,
    #     shuffle=False
    # )
