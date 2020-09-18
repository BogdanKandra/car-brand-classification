'''
Created on Fri Sep 18 11:58:07 2020

@author: Bogdan
'''
import os
import shutil

# Constants
ORIGINAL_DATASET_LOCATION = os.path.join(os.path.dirname(os.getcwd()), 'Data', 'Cars')
REORGANIZED_DATASET_LOCATION = 'dataset'

# Reorganize the original dataset into directories named after the car brands,
# containing all images representing a car brand, renamed and indexed by their
# model and year
is_dir_predicate = lambda path: os.path.isdir(os.path.join(ORIGINAL_DATASET_LOCATION, path))
dataset_directories = list(filter(is_dir_predicate, os.listdir(ORIGINAL_DATASET_LOCATION)))
car_brands = set([directory.split('_')[0] for directory in dataset_directories])

# For each car brand, collect all images representing that brand,
# rename them appropriately and copy them to a directory in the new location
for brand in car_brands:
    print('>>> Processing brand: {}...'.format(brand))
    brand_directories = [directory for directory in dataset_directories if directory.startswith(brand)]

    # Create directory corresponding to the current car brand in the new dataset location
    directory_path = os.path.join(REORGANIZED_DATASET_LOCATION, brand)
    try:
        os.mkdir(directory_path)
    except FileExistsError as err:
        # Delete the directory and all its contents and create it anew
        shutil.rmtree(directory_path, ignore_errors=True)
        os.mkdir(directory_path)

    for brand_directory in brand_directories:
        print('>>>>> Processing directory: {}...'.format(brand_directory))
        brand_directory_path = os.path.join(ORIGINAL_DATASET_LOCATION, brand_directory)
        image_names = os.listdir(brand_directory_path)

        image_index = 0
        for image_name in image_names:
            image_extension = image_name.split('.')[-1]
            original_image_path = os.path.join(brand_directory_path, image_name)
            new_image_name = brand_directory + '_' + str(image_index) + '.' + image_extension
            new_image_path = os.path.join(directory_path, new_image_name)
            shutil.copyfile(original_image_path, new_image_path)
            image_index += 1
