'''
Created on Fri Oct  9 16:57:51 2020

@author: Bogdan
'''
from collections import OrderedDict
import os
import shutil
import time
import utils

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
        print('>>> Processing brand: {}...'.format(brand))
        brand_directories = [directory for directory in dataset_directories if directory.startswith(brand)]

        # Create directory corresponding to the current car brand in the new dataset location
        directory_path = os.path.join(utils.DATASET_LOCATION, brand)
        os.mkdir(directory_path)

        for brand_directory in brand_directories:
            print('>>>>> Processing directory: {}...'.format(brand_directory))
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
    from each model and year are computed. These information are ploted and
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
        print('> Computing sample counts for brand: {}...'.format(car_brand))
        images_names = os.listdir(os.path.join(utils.DATASET_LOCATION, car_brand))
        all_model_names = ['_'.join(name.split('_')[1:-2]) for name in images_names]
        all_model_years = [name.split('_')[-2] for name in images_names]

        models = set(all_model_names)
        for car_model in models:
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
    utils.write_dictionary(sorted_classes_counts, 'all_brands_samples_counts.txt')
    utils.write_dictionary(top10_classes_counts, 'top_10_brands_samples_counts.txt')
    utils.write_dictionary(samples_information, 'top_10_brands_samples_information.txt')




##### Algorithm
if __name__ == '__main__':
    # # Create the reorganized dataset structure ('dataset' directory)
    # print('>>> Reorganizing the dataset and creating "dataset" directory...')
    # start_acquisition = time.time()
    # data_acquisition()
    # end_acquisition = time.time()
    # print('>>> Reorganizing the dataset took {}'.format(end_acquisition - start_acquisition))

    # # Analyze the dataset
    # print('>>> Analyzing the dataset...')
    # start_analysis = time.time()
    # data_analysis(save_plots=True)
    # end_analysis = time.time()
    # print('>>> Analyzing the dataset took {}'.format(end_analysis - start_analysis))

    pass
