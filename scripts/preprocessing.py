'''
Created on Fri Oct  9 16:57:51 2020

@author: Bogdan

This script performs the dataset reorganization, preprocessing and analysis
required for training the image classifier
'''
from collections import OrderedDict
import logging
import os
import shutil
import time
import utils
import numpy as np
from PIL import Image


### Set logging level and define logger
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

### Functions
def data_acquisition():
    ''' Reorganizes the original dataset, which was organized into 9170
    directories named following the pattern <carBrand_carModel_modelYear>. Since
    we are classifying cars by their brand, we reorganize the dataset into 75
    directories, named by car brands and rename images to follow the pattern
    <carBrand_carModel_modelYear_indexNumber>; so that performing a stratified
    split of the data into training and testing sets is possible
    '''
    # Determine the set of car brands
    is_dir_predicate = lambda path: os.path.isdir(os.path.join(utils.ORIGINAL_DATASET_LOCATION, path))
    dataset_directories = list(filter(is_dir_predicate, os.listdir(utils.ORIGINAL_DATASET_LOCATION)))
    car_brands = set([directory.split('_')[0] for directory in dataset_directories])

    # Create the directory corresponding to the reorganized dataset location
    try:
        os.mkdir(utils.DATASET_LOCATION)
    except FileExistsError:
        # Delete the directory and all of its contents and create it anew
        shutil.rmtree(utils.DATASET_LOCATION, ignore_errors=True)
        os.mkdir(utils.DATASET_LOCATION)

    # For each car brand, collect all images representing that brand,
    # rename them appropriately and copy them to a directory in the new location
    for brand in car_brands:
        LOGGER.info('> Processing brand: {}...'.format(brand))
        brand_directories = [directory for directory in dataset_directories if directory.startswith(brand)]

        # Create directory corresponding to the current car brand in the new dataset location
        directory_path = os.path.join(utils.DATASET_LOCATION, brand)
        os.mkdir(directory_path)

        for brand_directory in brand_directories:
            LOGGER.info('>> Processing directory: {}...'.format(brand_directory))
            brand_directory_path = os.path.join(utils.ORIGINAL_DATASET_LOCATION, brand_directory)
            image_names = os.listdir(brand_directory_path)

            image_index = 0
            for image_name in image_names:
                image_extension = image_name.split('.')[-1]
                original_image_path = os.path.join(brand_directory_path, image_name)
                new_image_name = brand_directory + '_' + str(image_index) + '.' + image_extension
                new_image_path = os.path.join(directory_path, new_image_name)
                shutil.copyfile(original_image_path, new_image_path)
                image_index += 1

def data_analysis(save_plots=False):
    ''' Performs data analysis. The top 10 car classes are determined, in terms
    of number of samples, and for each of the top classes, the number of samples
    from each model and year are computed. These information are plotted and
    written to files
    '''
    # Determine the top 10 classes
    is_dir_predicate = lambda path: os.path.isdir(os.path.join(utils.DATASET_LOCATION, path))
    dataset_directories = list(filter(is_dir_predicate, os.listdir(utils.DATASET_LOCATION)))
    classes_counts = {directory: len(os.listdir(os.path.join(utils.DATASET_LOCATION, directory))) for directory in dataset_directories}
    sorted_classes_counts = OrderedDict({k: v for k, v in sorted(classes_counts.items(), key=lambda item: item[1], reverse=True)})
    top10_classes_counts = OrderedDict({k: v for k, v in list(sorted_classes_counts.items())[:10]})

    # For each of the top 10 car classes, compute the number of samples from each
    # model and release year, as a dictionary. Will be used for performing easy
    # train-test-split stratification of the data
    samples_information = {}

    for car_brand in top10_classes_counts.keys():
        LOGGER.info('> Computing sample counts for brand: {}...'.format(car_brand))
        images_names = os.listdir(os.path.join(utils.DATASET_LOCATION, car_brand))
        all_model_names = ['_'.join(name.split('_')[1:-2]) for name in images_names]
        all_model_years = [name.split('_')[-2] for name in images_names]

        car_models = set(all_model_names)
        for car_model in car_models:
            years = set([all_model_years[i] for i in range(len(images_names)) if car_model == all_model_names[i]])
            for year in years:
                samples_count = len([i for i in range(len(images_names)) if car_model == all_model_names[i] and year == all_model_years[i]])
                key = car_brand + '|' + car_model + '|' + year
                samples_information[key] = samples_count

    samples_information = OrderedDict({k: v for k, v in sorted(samples_information.items(), key=lambda item: item[0])})

    # Plot the classes and their sample counts
    if save_plots is True:
        utils.save_bar_plot('All Classes Counts', 'Class Name', 'Sample Count', list(sorted_classes_counts.keys()), sorted_classes_counts.values(), 'b', 'All_Classes_Counts')
        utils.save_bar_plot('Top 10 Classes Counts', 'Class Name', 'Sample Count', list(top10_classes_counts.keys()), top10_classes_counts.values(), 'b', 'Top_10_Classes_Counts')
    else:
        utils.show_bar_plot(0, 'All Classes Counts', 'Class Name', 'Sample Count', list(sorted_classes_counts.keys()), sorted_classes_counts.values(), 'b')
        utils.show_bar_plot(1, 'Top 10 Classes Counts', 'Class Name', 'Sample Count', list(top10_classes_counts.keys()), top10_classes_counts.values(), 'b')

    # Write analysis information to files
    utils.write_dictionary(sorted_classes_counts, utils.ALL_BRANDS_COUNTS_NAME)
    utils.write_dictionary(top10_classes_counts, utils.TOP10_BRANDS_COUNTS_NAME)
    utils.write_dictionary(samples_information, utils.TOP10_BRANDS_INFORMATION_NAME)

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
    samples_information = utils.read_dictionary(utils.TOP10_BRANDS_INFORMATION_NAME)

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

def preparation_helper(training_set_subdirectory, class_name, file_names):
    ''' Helper function which copies and resizes the files specified in
    file_names, from the original dataset directory to the specified subset of
    the training data directory and class name directory '''
    for file_name in file_names:
        original_image_path = os.path.join(utils.DATASET_LOCATION, class_name, file_name)
        new_image_path = os.path.join(training_set_subdirectory, class_name, file_name)

        if os.path.exists(original_image_path + '.jpg'):
            extension = '.jpg'
        else:
            extension = '.png'

        original_image_path += extension
        new_image_path += extension

        # Load the image from the original path, resize it and save the resulting image
        Image.open(original_image_path) \
             .convert('RGB') \
             .resize((utils.RESIZE_WIDTH, utils.RESIZE_HEIGHT), Image.BICUBIC) \
             .save(new_image_path)

def prepare_training_dataset(train_names, test_names, delete_dataset=False):
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

        LOGGER.info('> Preparing train images for {}...'.format(class_name))
        train_files = [train_name for train_name in train_names if train_name.startswith(class_name)]
        preparation_helper(utils.TRAIN_SET_LOCATION, class_name, train_files)

        LOGGER.info('> Preparing test images for {}...'.format(class_name))
        test_files = [test_name for test_name in test_names if test_name.startswith(class_name)]
        preparation_helper(utils.TEST_SET_LOCATION, class_name, test_files)

    # Delete the original dataset, if required
    if delete_dataset is True:
        shutil.rmtree(utils.DATASET_LOCATION)

def subsample_data(data_percentage, random_state=None):
    ''' Randomly picks <data_percentage> % of the data specified in the
    top_brands_samples_information dictionary and loads them for fitting the
    Keras ImageDataGenerator (using the same preprocessing as used on the
    training data)

    Arguments:
        *data_percentage* (float) -- specifies the percentage of the data to be
        chosen for subsampling; value must be between 0 and 1

        *random_state* (int) -- specifies the seed to be used with the
        RandomState instance, so that the results are reproducible

    Returns:
        NumPy array -- the subsampled data
    '''
    # Initialize necessary variables
    np.random.seed(random_state)
    images, image_names = [], []

    # Stratify the data by brand, model and year
    samples_information = utils.read_dictionary(utils.TOP10_BRANDS_INFORMATION_NAME)

    LOGGER.info('> Randomly choosing files for subsampling...')
    for key in samples_information.keys():
        brand, model, year = key.split('|')
        count = samples_information[key]
        subsampling_count = int(data_percentage * count) if count > 39 else 1

        # Generate indices permutation for selecting subsample data
        available_indices = np.random.permutation(count)
        subsampling_indices = available_indices[:subsampling_count]

        # Generate and append the relevant image names
        image_base_name = brand + '_' + model + '_' + year + '_'
        key_subsample_image_names = [image_base_name + str(index) for index in subsampling_indices]
        image_names.extend(key_subsample_image_names)

    LOGGER.info('> Loading the images...')
    loaded_images = 0
    tenth_of_images = len(image_names) // 10
    for image_name in image_names:
        class_name = image_name[:image_name.index('_')]
        image_path = os.path.join(utils.DATASET_LOCATION, class_name, image_name)
        if os.path.exists(image_path + '.jpg'):
            extension = '.jpg'
        else:
            extension = '.png'
        image_path += extension

        image = Image.open(image_path) \
                     .convert('RGB') \
                     .resize((utils.RESIZE_WIDTH, utils.RESIZE_HEIGHT), Image.BICUBIC)
        images.append(np.array(image))
        loaded_images += 1

        if loaded_images % tenth_of_images == 0:
            LOGGER.info('> Loaded {}% of images'.format(loaded_images // tenth_of_images * 10))

    return np.array(images)



##### Algorithm
if __name__ == '__main__':

    # Only run the preprocessing steps if the "dataset" directory is not present
    if os.path.isdir(utils.DATASET_LOCATION) is False:

        # Create the reorganized dataset structure ("dataset" directory)
        LOGGER.info('>>> Reorganizing the dataset and creating "dataset" directory...')
        start_acquisition = time.time()
        data_acquisition()
        end_acquisition = time.time()
        LOGGER.info('>>> Reorganizing the dataset took {}\n'.format(end_acquisition - start_acquisition))

        # Analyze the dataset and save the results in the "figures" and "texts" directories
        LOGGER.info('>>> Analyzing the dataset...')
        start_analysis = time.time()
        data_analysis(save_plots=True)
        end_analysis = time.time()
        LOGGER.info('>>> Analyzing the dataset took {}\n'.format(end_analysis - start_analysis))

        # Split the names of the data files into training and testing sets
        LOGGER.info('>>> Splitting the data into training and testing sets...')
        start_split = time.time()
        train_image_names, test_image_names, y_train, y_test = train_test_split(utils.TRAIN_SET_PERCENTAGE, utils.RANDOM_STATE)
        end_split = time.time()
        LOGGER.info('>>> Splitting took {}\n'.format(end_split - start_split))

        # Create the training data directory structure for loading in Keras and resize images
        LOGGER.info('>>> Preparing the training dataset...')
        start_preparation = time.time()
        prepare_training_dataset(train_image_names, test_image_names)
        end_preparation = time.time()
        LOGGER.info('>>> Preparing the training dataset took {}\n'.format(end_preparation - start_preparation))
    else:
        LOGGER.info('>>> "dataset" directory already present... skipping data preprocessing')

    # Only perform the training data subsampling if the "pickles" directory is not present
    if os.path.isdir(utils.PICKLES_LOCATION) is False:
        os.mkdir(utils.PICKLES_LOCATION)

        # Subsample the training dataset for computing statistics necessary for preprocessing
        LOGGER.info('>>> Subsampling the training dataset...')
        start_subsampling = time.time()
        X_sample = subsample_data(utils.SUBSAMPLE_PERCENTAGE, utils.RANDOM_STATE)
        end_subsampling = time.time()
        LOGGER.info('>>> Subsampling took {}\n'.format(end_subsampling - start_subsampling))

        # Pickle the subsampled data for later uploading on Google Drive
        LOGGER.info('>>> Serializing the subsampled data...')
        start_serializing = time.time()
        utils.save_numpy_array(X_sample, utils.SUBSAMPLE_ARRAY_NAME)
        end_serializing = time.time()
        LOGGER.info('>>> Serializing took {}\n'.format(end_serializing - start_serializing))
    else:
        LOGGER.info('>>> "pickles" directory already present... loading saved data subsample')
        X_sample = utils.load_numpy_array(utils.SUBSAMPLE_ARRAY_NAME)
