# process_video.py
import cv2
import numpy as np
from moviepy import VideoFileClip
import pandas as pd
from tqdm import tqdm
import tensorflow as tf

# Assign Keras utilities and layers
load_model = tf.keras.models.load_model

# Load the trained model
model_path = r'D:\\projects\\emotion detection\\trial 2 (own model)\\emotion_model.h5'
model = load_model(model_path)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Function to preprocess frame for the model
def preprocess_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (48, 48))
    normalized_frame = resized_frame.astype('float32') / 255.0
    input_frame = normalized_frame.reshape(1, 48, 48, 1)
    return input_frame

# Function to detect emotions and return probabilities
def detect_emotions(frame):
    try:
        input_frame = preprocess_frame(frame)
        predictions = model.predict(input_frame, verbose=0)[0]
        emotion_probabilities = {emotion_labels[i]: float(predictions[i]) * 100 for i in range(len(emotion_labels))}
        return emotion_probabilities
    except Exception as e:
        print(f"Error analyzing frame: {e}")
        return None

# Function to calculate percentages of emotions
def calculate_emotion_percentages(emotion_probabilities):
    total_probability = sum(emotion_probabilities.values())
    if total_probability == 0:
        return {emotion: 0 for emotion in emotion_probabilities.keys()}
    emotion_percentages = {emotion: (prob / total_probability) * 100 for emotion, prob in emotion_probabilities.items()}
    return emotion_percentages

# Load your video
video_path = r'D:\\projects\\emotion detection\\videoplayback.mp4'
clip = VideoFileClip(video_path)

# List to hold emotion data for each frame
emotion_data = []

# Loop over video frames
for frame_index, frame in tqdm(enumerate(clip.iter_frames()), total=clip.reader.n_frames):
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    emotion_probabilities = detect_emotions(frame_bgr)
    
    if emotion_probabilities:
        emotion_percentages = calculate_emotion_percentages(emotion_probabilities)
        time_in_seconds = frame_index / clip.fps
        emotion_data.append({
            'frame_number': frame_index,
            'timeframe': time_in_seconds,
            **emotion_percentages
        })

# Convert the list of emotion data to a DataFrame
df = pd.DataFrame(emotion_data)

# Save the DataFrame to a CSV file
csv_file_path = r'D:\\projects\\emotion detection\\trial 2 (own model)\\emotion_data.csv'
df.to_csv(csv_file_path, index=False)

print(f"Emotion data saved to {csv_file_path}")