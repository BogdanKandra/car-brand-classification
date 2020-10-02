'''
Created on Fri Sep 18 10:20:51 2020

@author: Bogdan

This script performs data analysis. The top 10 car classes are determined, in
terms of number of samples, and for each of the top classes, the number of
samples from each model and year are computed. These information are ploted
and written to files
'''
from collections import OrderedDict
import os
import utils

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
utils.save_bar_plot('All Classes Counts', 'Class Name', 'Sample Count', list(sorted_classes_counts.keys()), sorted_classes_counts.values(), 'b', 'All_Classes_Counts')
utils.save_bar_plot('Top 10 Classes Counts', 'Class Name', 'Sample Count', list(top10_classes_counts.keys()), top10_classes_counts.values(), 'b', 'Top_10_Classes_Counts')
# utils.show_bar_plot(0, 'All Classes Counts', 'Class Name', 'Sample Count', list(sorted_classes_counts.keys()), sorted_classes_counts.values(), 'b')
# utils.show_bar_plot(1, 'Top 10 Classes Counts', 'Class Name', 'Sample Count', list(top10_classes_counts.keys()), top10_classes_counts.values(), 'b')

# Write analysis information to files
utils.write_dictionary(sorted_classes_counts, 'all_brands_samples_counts.txt')
utils.write_dictionary(top10_classes_counts, 'top_10_brands_samples_counts.txt')
utils.write_dictionary(samples_information, 'top_10_brands_samples_information.txt')
