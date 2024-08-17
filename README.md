# LFGuard: A Defense against Label Flipping Attack in Federated Learning for Vehicular Network
Code for the paper: LFGuard: A Defense against Label Flipping Attack in Federated Learning for Vehicular Network

**Installation**

1. Create a virtualenv (Python 3.7)
2. Install dependencies inside of virtualenv (pip install -r requirements.pip)

**Instructions for execution**

**Setup**

Before starting the experiments, you have to create the file sheets, logs, confusion_matrix, misclassification

Before you can run any experiments, you must complete some setup:
1.	python3 generate_data_distribution.py This downloads the datasets, as well as generates a static distribution of the training and test data to provide consistency in experiments. 
2.	python3 generate_default_models.py This generates an instance of all of the models used in the paper and saves them to disk.
3.	Python3 defense.py to start the running

