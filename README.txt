CREATING AN ENVIRONMENT FOR RUNNING THE PROJECT:
=============================
-> Create environment using: conda create --name car-brand-classification python=3.8
	-> Python version 3.8 is required since TensorFlow doesn't currently support Python versions newer than 3.8
-> Activate environment using: conda activate car-brand-classification
-> Install pip in the newly created environment: conda install pip
-> Install dependencies from requirements.txt: pip install -r requirements.txt

DIRECTORY STRUCTURE:
=============================
-> The project directory must be named 'car-brand-classification'
-> The original dataset directory must be located in the parent directory of the project directory,
in a directory named 'Data' containing a directory named 'Cars'
-> The directory structure should look like this:
	\car-brand-classification
	\Data\Cars\[dataset_directories]

RUNNING THE PROJECT:
=============================
-> Run "preprocessing.py" for:
	-> Reorganizing and analysing the dataset
	-> Splitting the data into training and testing sets
	-> Performing preprocessing on the training and testing sets
	-> A subsample is also computed and serialized, so that the Keras image data generator will be able to normalize the input images for training
-> Upload the newly obtained "training_data" and "pickles" directories to a Google Drive account
-> Upload the "notebooks" directory to the same Google Drive account
-> Run "notebooks\training.ipynb" in Google Colab for:
	-> Performing the training process and saving results
