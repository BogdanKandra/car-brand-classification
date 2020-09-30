'''
Created on Fri Sep 18 11:58:07 2020

@author: Bogdan

This script reorganizes the original dataset, which was organized into 9170
directories named following the pattern <carBrand_carModel_modelYear>. Since
we are classifying cars by their brand, we reorganize the dataset into 75
directories, named by car brands and rename images to follow the pattern
<carBrand_carModel_modelYear_indexNumber>; so that performing a stratified
split of the data into training and testing sets is possible
'''
import os
import shutil

# Constants
ORIGINAL_DATASET_LOCATION = os.path.join(os.path.dirname(os.getcwd()), 'Data', 'Cars')
REORGANIZED_DATASET_LOCATION = 'dataset'

# Determine the set of car brands
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
    except FileExistsError:
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
