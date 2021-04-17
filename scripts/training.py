'''
Created on Sat Jan  9 13:08:51 2021

@author: Bogdan

This script performs the training of the image classifier
'''
import logging
import matplotlib.pyplot as plt
import os
import time
import utils
import tensorflow as tf
import tensorflow.keras.applications as apps
from tensorflow.keras import layers, losses, metrics, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


### Set logging level and define logger
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

### Functions
def build_model_pooling_dropout(module_name, network_name):
    ''' Builds the model, starting from a base model specified by the module
    name and network name '''
    # Define layers
    preprocess_input = utils.load_input_preprocessing_function(module_name)
    base_model = utils.load_pretrained_network(network_name)
    avg_pooling_layer = layers.GlobalAveragePooling2D(name='avg_pooling_layer')
    # max_pooling_layer = layers.GlobalMaxPooling2D(name='max_pooling_layer')
    dropout_layer = layers.Dropout(0.5, name='dropout_layer')
    classification_layer = layers.Dense(10, activation='softmax', name='classification_layer')

    # Define model structure
    inputs = tf.keras.Input(shape=(utils.RESIZE_HEIGHT, utils.RESIZE_WIDTH, 3))
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = avg_pooling_layer(x)
    x = dropout_layer(x)
    outputs = classification_layer(x)
    model = tf.keras.Model(inputs, outputs)

    return model

def build_model_flatten_dense(module_name, network_name):
    ''' Builds the model, starting from a base model specified by the module
    name and network name '''
    # Define test layers
    preprocess_input = utils.load_input_preprocessing_function(module_name)
    base_model = utils.load_pretrained_network(network_name)
    flatten_layer = layers.Flatten(name='flatten')
    specialisation_layer = layers.Dense(1024, activation='relu', name='specialisation_layer')
    classification_layer = layers.Dense(10, activation='softmax', name='classification_layer')

    # Define test model
    inputs = tf.keras.Input(shape=(utils.RESIZE_HEIGHT, utils.RESIZE_WIDTH, 3))
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = flatten_layer(x)
    x = specialisation_layer(x)
    outputs = classification_layer(x)
    model = tf.keras.Model(inputs, outputs)

    return model

def plot_results(training_history):
    # Plot training and validation accuracy and loss
    training_accuracy = training_history['accuracy']
    validation_accuracy = training_history['val_accuracy']
    training_loss = training_history['loss']
    validation_loss = training_history['val_loss']

    plt.figure(figsize=(18, 10))
    plt.subplot(2, 1, 1)
    plt.plot(training_accuracy, label='Training Accuracy')
    plt.plot(validation_accuracy, label='Validation Accuracy')
    plt.legend()
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(training_loss, label='Training Loss')
    plt.plot(validation_loss, label='Validation Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Categorical Cross Entropy')
    plt.ylim([0, max(plt.ylim())])
    plt.title('Training and Validation Loss')

    if os.path.isdir(utils.TRAINING_RESULTS_FIGURES_LOCATION) is False:
        os.mkdir(utils.TRAINING_RESULTS_FIGURES_LOCATION)

    figure_path = os.path.join(utils.TRAINING_RESULTS_FIGURES_LOCATION, 'Training Results.png')
    plt.savefig(figure_path, quality=100)
    plt.close()


##### Algorithm
if __name__ == '__main__':

    # Load the necessary data into memory
    # X_sample = utils.load_numpy_array(utils.SUBSAMPLE_ARRAY_NAME)
    samples_counts = utils.read_dictionary(utils.TOP10_BRANDS_COUNTS_NAME)

    # # Create necessary directories
    # if os.path.isdir(utils.AUGMENTED_DIR) is False:
    #     os.mkdir(utils.AUGMENTED_DIR)
    #     os.mkdir(utils.TRAIN_AUGMENT_LOCATION)
    #     os.mkdir(utils.VALIDATION_AUGMENT_LOCATION)
    #     os.mkdir(utils.TEST_AUGMENT_LOCATION)

    # Create Keras data generators and iterators
    LOGGER.info('>>> Defining and Fitting the data generator...')
    start = time.time()
    
    # The augmentation is the same for all data sets, so a single generator is used
    data_generator = ImageDataGenerator(
        # featurewise_center=True,
        # featurewise_std_normalization=True
    )
    # data_generator.fit(X_sample)
    # del X_sample

    end = time.time()
    LOGGER.info('>>> Fitting the data generator took {}\n'.format(end - start))

    LOGGER.info('>>> Defining the data iterators...')
    start = time.time()

    train_iterator = data_generator.flow_from_directory(
        directory=utils.TRAIN_SET_LOCATION,
        target_size=(utils.RESIZE_HEIGHT, utils.RESIZE_WIDTH),
        color_mode='rgb',
        classes=list(samples_counts.keys()),
        class_mode='categorical',
        batch_size=utils.BATCH_SIZE,
        shuffle=True,
        # seed=utils.RANDOM_STATE,
        # save_to_dir=utils.TRAIN_AUGMENT_LOCATION,
        interpolation='bilinear'
    )

    validation_iterator = data_generator.flow_from_directory(
        directory=utils.VALIDATION_SET_LOCATION,
        target_size=(utils.RESIZE_HEIGHT, utils.RESIZE_WIDTH),
        color_mode='rgb',
        classes=list(samples_counts.keys()),
        class_mode='categorical',
        batch_size=utils.BATCH_SIZE,
        shuffle=False,
        # seed=utils.RANDOM_STATE,
        # save_to_dir=utils.VALIDATION_AUGMENT_LOCATION,
        interpolation='bilinear'
    )

    test_iterator = data_generator.flow_from_directory(
        directory=utils.TEST_SET_LOCATION,
        target_size=(utils.RESIZE_HEIGHT, utils.RESIZE_WIDTH),
        color_mode='rgb',
        classes=list(samples_counts.keys()),
        class_mode='categorical',
        batch_size=utils.BATCH_SIZE,
        shuffle=False,
        # seed=utils.RANDOM_STATE,
        # save_to_dir=utils.TEST_AUGMENT_LOCATION,
        interpolation='bilinear'
    )

    end = time.time()
    LOGGER.info('>>> Defining the iterators took {}\n'.format(end - start))

    # X_batch, y_batch = train_iterator.next()

    # Build the model
    module_name = 'mobilenet_v2'
    network_name = 'MobileNetV2'
    model = build_model_flatten_dense(module_name, network_name)
    # model = build_model_pooling_dropout(module_name, network_name)
    model.summary()

    # Define train parameters
    train_steps = len(train_iterator)
    validation_steps = len(validation_iterator)
    evaluation_steps = len(test_iterator)
    base_learning_rate = 0.00001 # TODO - scheduler for learning rate
    optimizer = optimizers.Adam(learning_rate=base_learning_rate)
    loss_function = losses.CategoricalCrossentropy()
    train_metrics = [metrics.CategoricalAccuracy(), metrics.AUC(), metrics.Precision(), metrics.Recall()]

    model.compile(optimizer=optimizer,
                  loss=loss_function,
                  metrics=train_metrics)

    LOGGER.info('>>> Training the model...')
    # initial_results = model.evaluate(test_iterator, steps=evaluation_steps,
    #                                  return_dict=True)
    # LOGGER.info('>>>>> Initial Results: {}'.format(initial_results))

    start = time.time()
    training_history = model.fit(train_iterator, epochs=20, verbose=1,
                                 validation_data=validation_iterator,
                                 callbacks=[],
                                 steps_per_epoch=train_steps,
                                 validation_steps=validation_steps)
    history = training_history.history
    end = time.time()
    LOGGER.info('>>> Training the model took {}\n'.format(end - start))

    LOGGER.info('>>> Evaluating the model...')
    start = time.time()
    final_results = model.evaluate(test_iterator, steps=evaluation_steps,
                                  return_dict=True)
    LOGGER.info('>>>>> Final Results: {}'.format(final_results))
    end = time.time()
    LOGGER.info('>>> Evaluating the model took {}\n'.format(end - start))

    plot_results(history)
