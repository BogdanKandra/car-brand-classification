'''
Created on Mon Mar 22 16:28:51 2021

@author: Bogdan

This script performs the selection of images used for training the Teachable
Machine model. We have 134862 images in the training set; Teachable Machine
doesn't support loading much more than 50000 images, so 40% of the data will be
picked for upload
'''
import logging
import os
import shutil
import numpy as np
import utils


### Set logging level and define logger
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

### Functions
def copy_images(training_set_subdirectory, file_names):
    ''' Helper function which copies the files specified in file_names, from
    the training_data subdirectory to the corresponding teachable_machine_data
    subdirectory '''
    for file_name in file_names:
        original_image_path = os.path.join(utils.TRAIN_SET_LOCATION, training_set_subdirectory, file_name)
        new_image_path = os.path.join(utils.TEACHABLE_MACHINE_DIR, training_set_subdirectory, file_name)
        shutil.copyfile(original_image_path, new_image_path)

def prepare_dataset_imbalanced(data_percentage=0.4, random_state=None):
    ''' Randomly picks <data_percentage> % of the data in the training set and
    and copies them to a separate directory, to be used directly for upload to
    Teachable Machine

    Arguments:
        *data_percentage* (float) -- specifies the percentage of the data to be
        chosen for upload to Teachable Machine; value must be between 0 and 1

        *random_state* (int) -- specifies the seed to be used with the
        RandomState instance, so that the results are reproducible

    Returns:
        NumPy array -- the subsampled data
    '''
    # Initialize necessary variables
    np.random.seed(random_state)
    samples_counts = utils.read_dictionary(utils.TOP10_BRANDS_COUNTS_NAME)

    # Create teachable machine dataset directory, if necessary
    if os.path.isdir(utils.TEACHABLE_MACHINE_DIR) is False:
        os.mkdir(utils.TEACHABLE_MACHINE_DIR)

    LOGGER.info('> Randomly choosing files for training Teachable Machine model...')
    for brand in samples_counts.keys():
        LOGGER.info('>>> Processing brand {}...'.format(brand))

        # Create subdirectory
        os.mkdir(os.path.join(utils.TEACHABLE_MACHINE_DIR, brand))

        # Compute number of images to copy
        brand_directory_path = os.path.join(utils.TRAIN_SET_LOCATION, brand)
        image_names = os.listdir(brand_directory_path)
        count = len(image_names)
        subsampling_count = int(data_percentage * count)

        # Select subsample image names
        available_indices = np.random.permutation(count)
        subsampling_indices = available_indices[:subsampling_count]
        relevant_image_names = [image_names[i] for i in subsampling_indices]

        # Copy the images from the source directory to a separate directory
        copy_images(brand, relevant_image_names)

def prepare_dataset_balanced(target_count=50000, random_state=None):
    ''' '''
    pass



##### Main algorithm
if __name__ == '__main__':
    # prepare_dataset_imbalanced(utils.TEACHABLE_MACHINE_TRAIN_SUBSAMPLE_PERCENTAGE,
    #                            utils.RANDOM_STATE)
    pass
