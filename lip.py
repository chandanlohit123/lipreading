import os
import cv2
import tensorflow as tf
import numpy as np
import imageio
import Levenshtein
from utils import num_to_char  # Importing num_to_char from a module named utils

# Function to load video frames
def load_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[190:236, 80:220, :])
    cap.release()
    
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std

# Function to load alignments
def load_alignments(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil':
            tokens = [*tokens, ' ', line[2]]
    return tokens

# Function to load data (frames and alignments)
def load_data(path):
    file_name = path.split('\\')[-1].split('.')[0]
    video_path = os.path.join('data', 's1', f'{file_name}.mpg')
    alignment_path = os.path.join('data', 'alignments', 's1', f'{file_name}.align')
    frames = load_video(video_path)
    alignments = load_alignments(alignment_path)
    return frames, alignments

# Function to calculate character-level accuracy
def calculate_accuracy(true_text, predicted_text):
    correct_chars = sum(1 for true_char, pred_char in zip(true_text, predicted_text) if true_char == pred_char)
    total_chars = max(len(true_text), len(predicted_text))
    accuracy = correct_chars / total_chars * 100
    return accuracy

# Function to calculate word error rate (WER)
def calculate_wer(true_text, predicted_text):
    return Levenshtein.distance(true_text.split(), predicted_text.split()) / len(true_text.split()) * 100

# Load LipNet model
import os
checkpoint_dir = "C:\\PROJECT\\LipNet\\models\\checkpoint"
# Load the model
model = tf.saved_model.load(checkpoint_dir)



# Load test data
test_path = '..\\data\\s1\\bbal6n.mpg'
frames, true_alignments = load_data(test_path)

# Make predictions
yhat = model.predict(tf.expand_dims(frames, axis=0))

# Decode predictions and true alignments
decoded_predicted_text = ''.join([x for x in num_to_char(np.argmax(yhat[0], axis=2)[0]).numpy() if x != ''])
true_text = ''.join(true_alignments)

# Calculate accuracy and WER
accuracy = calculate_accuracy(true_text, decoded_predicted_text)
wer = calculate_wer(true_text, decoded_predicted_text)

# Print results
print("Character-level accuracy:", accuracy, "%")
print("Word Error Rate (WER):", wer, "%")
