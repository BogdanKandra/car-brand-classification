CREATING AN ENVIRONMENT FOR RUNNING THE PROJECT:
=============================
-> Create environment using: conda create --name car-brand-classification python=3.8
	-> Python version 3.8 is required since TensorFlow doesn't currently support Python versions newer than 3.8
-> Activate environment using: conda activate car-brand-classification
-> Install pip in the newly created environment: conda install pip
-> Install dependencies from requirements.txt: pip install -r requirements.txt

DIRECTORY STRUCTURE:
=============================
- The original dataset directory must be located in the parent directory of the project directory,
in a directory named 'Data' containing a directory named 'Cars'. The directory structure should look like this:
	\car-brand-classification\...
	\Data\Cars\dataset_directories
