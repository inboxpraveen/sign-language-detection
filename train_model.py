# Import necessary libraries and functions
from function import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
import numpy as np
import os

MODEL_CONFIG_FILE = "model.json"   ## model2.json, model3.json, ...
MODEL_NAME = "model.keras"         ## model2.keras, model3.keras, ...
EPOCHS = 100                       ## Try values like 200, 300, 400

# Creating a map from labels to numbers for classification
label_map = {label:num for num, label in enumerate(actions)}

# Initialize lists to store sequences (X) and labels (y)
sequences, labels = [], []

# Loop over each action and each sequence within that action
for action in actions:
    for sequence in range(no_sequences):
        window = [] # Temporary list to store frames of a single sequence

        # Loop over each frame in the sequence
        for frame_num in range(sequence_length):
            # Load the frame and append it to the window
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)

        # Add the window (sequence of frames) and its label to the main lists
        sequences.append(window)
        labels.append(label_map[action])

# Convert sequences and labels to numpy arrays for ML processing
X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# Setting up TensorBoard for model performance visualization
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Define a Sequential model using LSTM layers
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 63)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# Compile the model with optimizer, loss function, and metrics
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train the model with the training data
model.fit(X_train, y_train, epochs=EPOCHS, callbacks=[tb_callback])

# Display the summary of the model
model.summary()

# Save the model in JSON format
model_json = model.to_json()
with open(MODEL_CONFIG_FILE, "w") as json_file:
    json_file.write(model_json)

# Save the model weights
model.save(MODEL_NAME)