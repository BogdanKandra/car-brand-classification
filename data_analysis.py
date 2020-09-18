'''
Created on Fri Sep 18 10:20:51 2020

@author: Bogdan
'''
from collections import OrderedDict
import matplotlib.pyplot as plt
import os

# Constants
DATASET_LOCATION = 'dataset'
FIGURES_LOCATION = 'figures'

# Determine the top 15 classes in terms of number of samples
is_dir_predicate = lambda path: os.path.isdir(os.path.join(DATASET_LOCATION, path))
dataset_directories = list(filter(is_dir_predicate, os.listdir(DATASET_LOCATION)))
classes_counts = {directory: len(os.listdir(os.path.join(DATASET_LOCATION, directory))) for directory in dataset_directories}
sorted_classes_counts = OrderedDict({k: v for k, v in sorted(classes_counts.items(), key=lambda item: item[1], reverse=True)})
top15_classes_counts = OrderedDict({k: v for k, v in list(sorted_classes_counts.items())[:15]})
top10_classes_counts = OrderedDict({k: v for k, v in list(sorted_classes_counts.items())[:10]})

# plt.figure(0, figsize=(19.2, 10.8))
# plt.title('All Classes Counts')
# plt.xlabel('Class Name', fontweight='bold')
# plt.ylabel('Sample Count', fontweight='bold')
# plt.bar(list(sorted_classes_counts.keys()), sorted_classes_counts.values(), color='b')
# plt.savefig(os.path.join(FIGURES_LOCATION, 'All_Classes_Counts.png'), quality=100)
# plt.close()

# plt.figure(1, figsize=(19.2, 10.8))
# plt.title('Top 15 Classes Counts')
# plt.xlabel('Class Name', fontweight='bold')
# plt.ylabel('Sample Count', fontweight='bold')
# plt.bar(list(top15_classes_counts.keys()), top15_classes_counts.values(), color='b')
# plt.savefig(os.path.join(FIGURES_LOCATION, 'Top_15_Classes_Counts.png'), quality=100)
# plt.close()
