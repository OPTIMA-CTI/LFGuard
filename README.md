# LFGuard
This work explores a defense strategy named LFGuard for the vehicular network, a server-side defense approach that allows the detection of malicious nodes in the vehicular network with varying adversarial capabilities under different attack scenarios

**Instructions for execution**
**Setup**
Before starting the experiments, you have to create the file sheets, logs, confusion_matrix, misclassification
Before you can run any experiments, you must complete some setup:
1.	python3 generate_data_distribution.py This downloads the datasets, as well as generates a static distribution of the training and test data to provide consistency in experiments.
2.	python3 generate_default_models.py This generates an instance of all of the models used in the paper and saves them to disk.
3.	Python3 defense.py to start the running
