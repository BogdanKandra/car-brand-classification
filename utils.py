'''
Created on Tue Sep 22 08:54:15 2020

@author: Bogdan
'''
import os
import matplotlib.pyplot as plt

# Constants
FIGURES_LOCATION = 'figures'

def save_bar_plot(title, xlabel, ylabel, xdata, ydata, color='r', plot_name='figure'):
    ''' Generates a bar plot using the given data '''
    plt.figure(0, figsize=(19.2, 10.8))
    plt.title(title)
    plt.xlabel(xlabel, fontweight='bold')
    plt.ylabel(ylabel, fontweight='bold')
    plt.bar(xdata, ydata, color=color)
    plt.savefig(os.path.join(FIGURES_LOCATION, plot_name + '.png'), quality=100)
    plt.close()
