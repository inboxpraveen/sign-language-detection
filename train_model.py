# # Import necessary libraries and functions
# from function import *
# from sklearn.model_selection import train_test_split
# from keras.utils import to_categorical
# from keras.models import Sequential
# from keras.layers import LSTM, Dense
# from keras.callbacks import TensorBoard
# import numpy as np
# import os

# MODEL_CONFIG_FILE = "model.json"   ## model2.json, model3.json, ...
# MODEL_NAME = "model.keras"         ## model2.keras, model3.keras, ...
# EPOCHS = 100                       ## Try values like 200, 300, 400

# import splitfolders
# dr = 'SignImage48x48'
# splitfolders.ratio(dr,"splitdataset48x48" ,ratio=(0.8,0.2))

# # Creating a map from labels to numbers for classification
# label_map = {label:num for num, label in enumerate(actions)}

# # Initialize lists to store sequences (X) and labels (y)
# sequences, labels = [], []

# # Loop over each action and each sequence within that action
# for action in actions:
#     for sequence in range(no_sequences):
#         window = [] # Temporary list to store frames of a single sequence

#         # Loop over each frame in the sequence
#         for frame_num in range(sequence_length):
#             # Load the frame and append it to the window
#             res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
#             window.append(res)

#         # Add the window (sequence of frames) and its label to the main lists
#         sequences.append(window)
#         labels.append(label_map[action])

# # Convert sequences and labels to numpy arrays for ML processing
# X = np.array(sequences)
# y = to_categorical(labels).astype(int)

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# # Setting up TensorBoard for model performance visualization
# log_dir = os.path.join('Logs')
# tb_callback = TensorBoard(log_dir=log_dir)

# # Define a Sequential model using LSTM layers
# model = Sequential()
# model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 63)))
# model.add(LSTM(128, return_sequences=True, activation='relu'))
# model.add(LSTM(64, return_sequences=False, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(actions.shape[0], activation='softmax'))

# # Compile the model with optimizer, loss function, and metrics
# model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# # Train the model with the training data
# model.fit(X_train, y_train, epochs=EPOCHS, callbacks=[tb_callback])

# # Display the summary of the model
# model.summary()

# # Save the model in JSON format
# model_json = model.to_json()
# with open(MODEL_CONFIG_FILE, "w") as json_file:
#     json_file.write(model_json)

# # Save the model weights
# model.save(MODEL_NAME)



from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
import os
from config import *

def create_datagen():
    return ImageDataGenerator(rescale=1./255)

def create_generator(datagen, directory, batch_size):
    return datagen.flow_from_directory(
        directory,
        target_size=(48, 48),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='grayscale'
    )

def build_model(class_count):
    model = Sequential([
        Conv2D(128, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.4),
        Conv2D(256, kernel_size=(3,3), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.4),
        Conv2D(512, kernel_size=(3,3), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.4),
        Conv2D(512, kernel_size=(3,3), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.4),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(class_count, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def save_model(model, directory):
    model_json = model.to_json()
    with open(f"{directory}/model.json", 'w') as json_file:
        json_file.write(model_json)
    model.save(f"{directory}/model.keras")

# Main execution
def main():
    batch_size = 4

    train_datagen = create_datagen()
    val_datagen = create_datagen()

    train_generator = create_generator(train_datagen, f'{TRAINING_DATA_DIRECTORY}/train', batch_size)
    validation_generator = create_generator(val_datagen, f'{TRAINING_DATA_DIRECTORY}/val', batch_size)

    class_names = list(train_generator.class_indices.keys())
    print(class_names)

    model = build_model(len(class_names))

    steps_per_epoch = train_generator.samples // batch_size
    validation_steps = validation_generator.samples // batch_size

    model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs = 100,
        validation_data=validation_generator,
        validation_steps=validation_steps
    )

    save_model(model, MODEL_DIRECTORY)

if __name__ == "__main__":
    main()
