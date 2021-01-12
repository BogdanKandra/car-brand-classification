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
-> If you want to retrain the algorithm, perform the following:
	-> Create and activate an environment as outlined above
	-> Run "scripts\preprocessing.py" for:
		-> Reorganizing and analysing the dataset
		-> Splitting the data into training and testing sets
		-> Performing preprocessing on the training and testing sets
		-> Computing and serializing a subsample, so that the Keras image data generator will be able to normalize the input images for training
	-> Upload the "notebooks" and "training_data" directories to a Google Drive account
	-> Upload the "pickles\subsample.npy" and "texts\top_10_brands_samples_counts.txt" files to a directory named "resources" to the same Google Drive account
	-> Run "notebooks\training.ipynb" in Google Colab for:
		-> Performing the training process and saving the results
