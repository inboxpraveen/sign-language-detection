from keras.models import model_from_json
import cv2
import numpy as np
import os
from config import *


def load_model():
    with open(f"{MODEL_DIRECTORY}/model.json", "r") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights(f"{MODEL_DIRECTORY}/model.keras")
    return model


def extract_features(image):
    feature = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.resize(feature, (48, 48))
    feature = np.array(feature).reshape(1, 48, 48, 1)
    return feature / 255.0


def main():
    model = load_model()
    cap = cv2.VideoCapture(0)
    labels = os.listdir(ORIGINAL_DATA_DIRECTORY)
    print(labels)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.rectangle(frame, (0, 40), (300, 300), (0, 165, 255), 1)
        cropped_frame = frame[40:300, 0:300]
        features = extract_features(cropped_frame)
        pred = model.predict(features)
        prediction_label = labels[pred.argmax()]
        display_text = " " if prediction_label == 'blank' else f'{prediction_label}  {np.max(pred)*100:.2f}%'
        
        cv2.rectangle(frame, (0, 0), (300, 40), (0, 165, 255), -1)
        cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("output", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Esc key to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
