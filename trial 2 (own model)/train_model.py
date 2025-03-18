 # train_model.py
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Assign Keras utilities and layers
to_categorical = tf.keras.utils.to_categorical
Sequential = tf.keras.models.Sequential
Conv2D = tf.keras.layers.Conv2D
MaxPooling2D = tf.keras.layers.MaxPooling2D
Dense = tf.keras.layers.Dense
Flatten = tf.keras.layers.Flatten
Dropout = tf.keras.layers.Dropout

# Define paths and emotion labels
base_path = r'D:\\projects\\emotion detection\\trial 2 (own model)\\fer2013'
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
num_classes = len(emotion_labels)

# Function to load images from a directory
def load_images_from_folder(folder_path, subset='train'):
    images = []
    labels = []
    
    subset_path = os.path.join(folder_path, subset)
    for label_idx, emotion in enumerate(emotion_labels):
        emotion_folder = os.path.join(subset_path, emotion)
        for img_name in os.listdir(emotion_folder):
            img_path = os.path.join(emotion_folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (48, 48))
                img = img.astype('float32') / 255.0
                img = np.expand_dims(img, axis=-1)  # Shape: (48, 48, 1)
                images.append(img)
                labels.append(label_idx)
    
    return np.array(images), np.array(labels)

# Load train and test data
print("Loading training data...")
X_train, y_train = load_images_from_folder(base_path, subset='train')
print("Loading test data...")
X_test, y_test = load_images_from_folder(base_path, subset='test')

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

# Split train into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}, Test samples: {X_test.shape[0]}")

# Build the CNN model
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')  # 7 emotion classes
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Create and train the model
model = build_model()
model.summary()

# Train the model
history = model.fit(X_train, y_train, 
                    validation_data=(X_val, y_val), 
                    epochs=30, 
                    batch_size=64)

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")

# Save the trained model
model.save(r'D:\\projects\\emotion detection\\trial 2 (own model)\\emotion_model.h5')
print("Model saved as 'emotion_model.h5'")