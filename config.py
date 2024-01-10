import os

ORIGINAL_DATA_DIRECTORY = "MP_Data"
TRAINING_DATA_DIRECTORY = "TrainingData"
MODEL_DIRECTORY = "Models"

if not os.path.exists(ORIGINAL_DATA_DIRECTORY):
    os.mkdir(ORIGINAL_DATA_DIRECTORY)
    
if not os.path.exists(TRAINING_DATA_DIRECTORY):
    os.mkdir(TRAINING_DATA_DIRECTORY)

if not os.path.exists(MODEL_DIRECTORY):
    os.makedirs(MODEL_DIRECTORY)

