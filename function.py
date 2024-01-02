# Import necessary libraries
import cv2
import numpy as np
import os
import mediapipe as mp

# Initialize MediaPipe solutions for drawing and hand detection
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def mediapipe_detection(image, model):
    """
    Processes an image using the MediaPipe model and returns the processed image and results.

    Parameters:
    image (ndarray): The image to process.
    model (MediaPipe Hands): The MediaPipe model to use for hand detection.

    Returns:
    tuple: The processed image and the detection results.
    """
    # Convert color from BGR to RGB, process the image, and convert back
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    """
    Draws landmarks and connections on the image if hands are detected.

    Parameters:
    image (ndarray): The image on which to draw.
    results (MediaPipe Hands Results): The results from hand detection.
    """
    # Draw hand landmarks and connections if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

def extract_keypoints(results):
    """
    Extracts keypoints from the MediaPipe detection results.

    Parameters:
    results (MediaPipe Hands Results): The results from hand detection.

    Returns:
    ndarray: A flattened array of keypoints.
    """
    # Extract and flatten hand landmarks keypoints if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            rh = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten() if hand_landmarks else np.zeros(21*3)
            return np.concatenate([rh])

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data') 

# Define actions and configurations for sequences
actions = np.array(['A', 'B', 'C'])
no_sequences = 30
sequence_length = 30