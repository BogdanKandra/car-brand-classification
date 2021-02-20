'''
Created on Sat Jan  9 13:08:51 2021

@author: Bogdan

This script performs the training of the image classifier
'''
import logging
import os
import time
import utils
import tensorflow.keras.applications as apps
from tensorflow.keras import layers, losses, metrics, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm.keras import TqdmCallback


### Set logging level and define logger
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

### Functions
def load_pretrained_network(network_name):
    ''' Loads the specified pretrained network from Keras applications, with
    frozen weights '''
    image_shape = (utils.RESIZE_HEIGHT, utils.RESIZE_WIDTH, 3)
    base_model = getattr(apps, network_name)(include_top=False, weights='imagenet', input_shape=image_shape)
    base_model.trainable = False

    return base_model



##### Algorithm
if __name__ == '__main__':

    # Load the data subsample into memory
    X_sample = utils.load_numpy_array(utils.SUBSAMPLE_ARRAY_NAME)

    # Create Keras data generators and iterators
    samples_counts = utils.read_dictionary(utils.TOP10_BRANDS_COUNTS_NAME)
    if os.path.isdir(utils.AUGMENTED_DIR) is False:
        os.mkdir(utils.AUGMENTED_DIR)
        os.mkdir(utils.TEST_AUGMENT_LOCATION)
        os.mkdir(utils.TRAIN_AUGMENT_LOCATION)

    LOGGER.info('>>> Defining and Fitting the Data Generator...')
    start = time.time()
    # The augmentation is the same for both train and test sets, so a single generator is used
    data_generator = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True
    )

    data_generator.fit(X_sample)
    end = time.time()
    del X_sample
    LOGGER.info('>>> Fitting the data generator took {}\n'.format(end - start))

    LOGGER.info('>>> Defining train iterator...')
    start = time.time()
    train_iterator = data_generator.flow_from_directory(
        directory=utils.TRAIN_SET_LOCATION,
        target_size=(utils.RESIZE_HEIGHT, utils.RESIZE_WIDTH), # Size of MobileNet inputs is (224, 224)
        color_mode='rgb',
        classes=list(samples_counts.keys()),
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=utils.RANDOM_STATE,
        # save_to_dir=utils.TRAIN_AUGMENT_LOCATION,
        interpolation='bilinear'
    )
    end = time.time()
    LOGGER.info('>>> Defining the train iterator took {}\n'.format(end - start))

    LOGGER.info('>>> Defining test iterator...')
    start = time.time()
    test_iterator = data_generator.flow_from_directory(
        directory=utils.TEST_SET_LOCATION,
        target_size=(utils.RESIZE_HEIGHT, utils.RESIZE_WIDTH), # Size of MobileNet inputs is (224, 224)
        color_mode='rgb',
        classes=list(samples_counts.keys()),
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=utils.RANDOM_STATE,
        # save_to_dir=utils.TEST_AUGMENT_LOCATION,
        interpolation='bilinear'
    )
    end = time.time()
    LOGGER.info('>>> Defining the test iterator took {}\n'.format(end - start))

    # X_batch, y_batch = train_iterator.next()

    # Define models, metrics and optimizers to be tested
    LOGGER.info('>>> Defining models, metrics and optimizers...')
    start = time.time()
    model_names = {
        'efficientnet': ['EfficientNetB0', 'EfficientNetB3', 'EfficientNetB7'],
        'mobilenet': ['MobileNet'],
        'mobilenet_v2': ['MobileNetV2'],
        'nasnet': ['NASNetMobile'],
        'resnet50': ['ResNet50'],
        'resnet_v2': ['ResNet50V2'],
        'vgg16': ['VGG16']
    }

    loss_function = losses.CategoricalCrossentropy()
    train_metrics = [metrics.Accuracy(), metrics.AUC(), metrics.Precision(), metrics.Recall()]
    train_optimizers = [optimizers.Adam(), optimizers.RMSprop(), optimizers.Adadelta()]

    end = time.time()
    LOGGER.info('>>> Defining the training objects took {}\n'.format(end - start))

    # # Perform training for each model and optimizer
    # for module in model_names.keys():
    #     for network in model_names[module]:
    #         base_model = load_pretrained_network(network)
    #         # Complete the model
    #         model = models.Sequential(
    #             [
    #                 base_model,
    #                 layers.Flatten(name='flatten'),
    #                 layers.Dense(1024, activation='relu', name='specialisation_layer'),
    #                 layers.Dense(10, activation='softmax', name='classification_layer')
    #             ]
    #         )
    #         # base - pooling - dropout (0.2) - dense (num_classes)
    #         # base - flatten - dense (256) - dropout (0.5) - dense (num_classes)
    #         # base - flatten - dense (1024) - dense (num_classes)

    #         for optimizer in train_optimizers:
    #             # Compile the model
    #             model.compile(optimizer=optimizer, loss=loss_function, metrics=train_metrics)
    #             # Fit the model
    #             training_history = model.fit(train_iterator, epochs=50, verbose=2)
    #             # Evaluate the model
    #             results = model.evaluate()

    base_model = load_pretrained_network('VGG16')
    model = models.Sequential(
        [
            base_model,
            layers.Flatten(name='flatten'),
            layers.Dense(1024, activation='relu', name='specialisation_layer'),
            layers.Dense(10, activation='softmax', name='classification_layer')
        ]
    )

    optimizer = optimizers.Adam()

    LOGGER.info('>>> Compiling the model...')
    start = time.time()
    model.compile(optimizer=optimizer, loss=loss_function, metrics=train_metrics)
    end = time.time()
    LOGGER.info('>>> Compiling the model took {}\n'.format(end - start))

    LOGGER.info('>>> Training the model...')
    start = time.time()
    training_history = model.fit(train_iterator, epochs=10, verbose=0, callbacks=[TqdmCallback(verbose=2)])
    end = time.time()
    LOGGER.info('>>> Training the model took {}\n'.format(end - start))

    LOGGER.info('>>> Evaluating the model...')
    start = time.time()
    results_dict = model.evaluate(test_iterator, return_dict=True, callbacks=[TqdmCallback(verbose=2)])
    end = time.time()
    LOGGER.info('>>> Evaluating the model took {}\n'.format(end - start))

    LOGGER.info('>>> Training results:\n{}\n'.format(training_history))
    LOGGER.info('>>> Evaluation results:\n{}\n'.format(results_dict))
