from function import *
from keras.models import model_from_json
import numpy as np
import cv2


MODEL_NAME = "model.keras"
MODEL_CONFIG_FILE = "model.json"

class SignLanguageDetector:
    def __init__(self):
        self.load_model()
        self.colors = [(245, 117, 16) for _ in range(20)]
        self.threshold = 0.8
        self.sequence = []
        self.sentence = []
        self.accuracy = []
        self.predictions = []
        self.initialize_video_capture()

    def load_model(self):
        with open(MODEL_CONFIG_FILE, "r") as json_file:
            model_json = json_file.read()
        self.model = model_from_json(model_json)
        self.model.load_weights(MODEL_NAME)

    def prob_viz(self, res, actions, input_frame):
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):
            cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), self.colors[num], -1)
            cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return output_frame

    def initialize_video_capture(self):
        self.cap = cv2.VideoCapture(0)

    def process_video(self):
        with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                cropframe = frame[40:400, 0:300]
                frame = cv2.rectangle(frame, (0, 40), (300, 400), 255, 2)
                image, results = mediapipe_detection(cropframe, hands)

                keypoints = extract_keypoints(results)
                self.sequence.append(keypoints)
                self.sequence = self.sequence[-30:]

                try:
                    if len(self.sequence) == 30:
                        res = self.model.predict(np.expand_dims(self.sequence, axis=0))[0]
                        print(actions[np.argmax(res)])
                        self.predictions.append(np.argmax(res))

                        if np.unique(self.predictions[-10:])[0] == np.argmax(res):
                            if res[np.argmax(res)] > self.threshold:
                                self.update_sentence(actions[np.argmax(res)], res[np.argmax(res)])

                        frame = self.update_frame(frame)
                except Exception as e:
                    pass

                cv2.imshow('OpenCV Feed', frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            self.cap.release()
            cv2.destroyAllWindows()

    def update_sentence(self, action, score):
        if len(self.sentence) > 0:
            if action != self.sentence[-1]:
                self.sentence.append(action)
                self.accuracy.append(str(score * 100))
        else:
            self.sentence.append(action)
            self.accuracy.append(str(score * 100))

        if len(self.sentence) > 1:
            self.sentence = self.sentence[-1:]
            self.accuracy = self.accuracy[-1:]

    def update_frame(self, frame):
        cv2.rectangle(frame, (0, 0), (300, 40), (245, 117, 16), -1)
        cv2.putText(frame, "Output: -" + ' '.join(self.sentence) + ''.join(self.accuracy), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return frame

if __name__ == "__main__":
    detector = SignLanguageDetector()
    detector.process_video()