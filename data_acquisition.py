'''
Created on Fri Sep 18 11:58:07 2020

@author: Idrive
'''
import os

# Constants
WORKING_DIRECTORY = os.getcwd()
ORIGINAL_DATASET_LOCATION = os.path.join(os.path.dirname(WORKING_DIRECTORY), 'Data', 'Cars')
REORGANIZED_DATASET_LOCATION = 'dataset'

# Reorganize the original dataset into directories named after the car brands, containing all images representing a car brand, renamed and indexed by their model and year
is_dir_predicate = lambda path: os.path.isdir(os.path.join(ORIGINAL_DATASET_LOCATION, path))
dataset_directories = list(filter(is_dir_predicate, os.listdir(ORIGINAL_DATASET_LOCATION)))
car_brands = set([directory.split('_')[0] for directory in dataset_directories])

# For each car brand, collect all images representing that brand, rename them appropriately and copy them to a directory in the new location
for brand in car_brands:
    print('>>> Processing brand {}...'.format(brand))
    brand_directories = [directory for directory in dataset_directories if directory.startswith(brand)]
    for brand_directory in brand_directories:
        print('>>>>> Processing brand directory {}...'.format(brand_directory))
        images = os.listdir(os.path.join(ORIGINAL_DATASET_LOCATION, brand_directory))
