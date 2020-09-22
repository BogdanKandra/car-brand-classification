'''
Created on Fri Sep 18 10:20:51 2020

@author: Bogdan
'''
from collections import OrderedDict
import os
import utils

# Constants
DATASET_LOCATION = 'dataset'

# Determine the top 10 classes in terms of number of samples
is_dir_predicate = lambda path: os.path.isdir(os.path.join(DATASET_LOCATION, path))
dataset_directories = list(filter(is_dir_predicate, os.listdir(DATASET_LOCATION)))
classes_counts = {directory: len(os.listdir(os.path.join(DATASET_LOCATION, directory))) for directory in dataset_directories}
sorted_classes_counts = OrderedDict({k: v for k, v in sorted(classes_counts.items(), key=lambda item: item[1], reverse=True)})
top10_classes_counts = OrderedDict({k: v for k, v in list(sorted_classes_counts.items())[:10]})

# Plot the classes and their sample counts
utils.save_bar_plot('All Classes Counts', 'Class Name', 'Sample Count', list(sorted_classes_counts.keys()), sorted_classes_counts.values(), 'b', 'All_Classes_Counts')
utils.save_bar_plot('Top 10 Classes Counts', 'Class Name', 'Sample Count', list(top10_classes_counts.keys()), top10_classes_counts.values(), 'b', 'Top_10_Classes_Counts')
