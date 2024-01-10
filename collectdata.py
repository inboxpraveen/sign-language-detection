import cv2
import os
import argparse
from config import *
import splitfolders

def create_directory(path):
    if not os.path.exists(path):
        os.mkdir(path)

def get_file_counts(directory, range_start='a', range_end='z'):
    return {chr(i).lower(): len(os.listdir(os.path.join(directory, chr(i)))) 
            for i in range(ord(range_start), ord(range_end) + 1)}

# Argument parser for optional range input
parser = argparse.ArgumentParser(description='Create directories for specified range.')
parser.add_argument('--range', type=str, help='Range of alphabets, e.g., "a-d"', default='a-z')
args = parser.parse_args()
range_start, range_end = args.range.split('-') if '-' in args.range else ('a', 'z')

directory = f'{ORIGINAL_DATA_DIRECTORY}/'
create_directory(directory)
create_directory(os.path.join(directory, 'blank'))

# Create directories within the specified range
for i in range(ord(range_start), ord(range_end) + 1):
    create_directory(os.path.join(directory, chr(i).upper()))

cap = cv2.VideoCapture(0)
count = get_file_counts(directory, range_start, range_end)
count['blank'] = 0

while True:
    _, frame = cap.read()
    cv2.rectangle(frame, (0, 40), (300, 300), (255, 255, 255), 2)
    cv2.imshow("data", frame)

    cropped_frame = frame[40:300, 0:300]
    cv2.imshow("ROI", cropped_frame)

    gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (48, 48))

    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27:  # Esc key to exit
        break
    elif ord(range_start) <= (interrupt & 0xFF) <= ord(range_end):  # within specified range
        key = chr(interrupt & 0xFF)
        cv2.imwrite(os.path.join(directory, key.upper(), str(count[key]) + '.jpg'), resized_frame)
        count[key] += 1
    elif interrupt & 0xFF == 46:  # '.' key for blank
        cv2.imwrite(os.path.join(directory, 'blank', str(count['blank']) + '.jpg'), resized_frame)
        count['blank'] += 1

cap.release()
cv2.destroyAllWindows()

splitfolders.ratio(f'{ORIGINAL_DATA_DIRECTORY}', f"{TRAINING_DATA_DIRECTORY}", ratio=(0.8, 0.2))

# Removing empty directories within the specified range
for i in range(ord(range_start), ord(range_end) + 1):
    if os.path.exists(os.path.join(directory, chr(i).upper())):
        if not os.listdir(os.path.join(directory, chr(i).upper())):
            os.remove(os.path.join(directory, chr(i).upper()))
