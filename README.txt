CREATING AN ENVIRONMENT FOR RUNNING THE PROJECT:
=============================
-> Create environment using: conda create car-brand-classification
-> Activate environment using: conda activate car-brand-classification
-> Install pip in the newly created environment: conda install pip
-> Install dependencies from requirements.txt: pip install -r requirements.txt

DIRECTORY STRUCTURE:
=============================
- The original dataset directory must be located in the parent directory of the project directory,
in a directory named 'Data' containing a directory named 'Cars'. The directory structure should look like this:
	\car-brand-classification\...
	\Data\Cars\dataset_directories

- Uncomment the first four code blocks and run them, for:
	- Creating the reorganized dataset structure ('dataset' directory)
	- Running an analysis on the dataset (writing files in 'figures' and 'texts' directories)
	- Creating the directory structure for loading the training data in Keras
