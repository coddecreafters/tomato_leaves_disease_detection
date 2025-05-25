import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image
import os
import requests

def create_model(input_shape=(224, 224, 3), num_classes=5):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def preprocess_image(image_path, target_size=(224, 224)):
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        img = img.resize(target_size)
        img_array = np.array(img) / 255.0

        if img_array.shape != (*target_size, 3):
            raise ValueError(f"Invalid image shape: {img_array.shape}")

        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None

def predict_disease(model, image_path):
    try:
        if model is None:
            raise ValueError("Model is not loaded")

        img_array = preprocess_image(image_path)
        if img_array is None:
            raise ValueError("Failed to preprocess image")

        predictions = model.predict(img_array, verbose=0)
        return predictions[0]
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        return None

DISEASE_CLASSES = [
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_healthy'
]

def download_model_from_huggingface(model_url, model_path):
    try:
        print(f"Downloading model from: {model_url}")
        response = requests.get(model_url)
        response.raise_for_status()

        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        with open(model_path, 'wb') as f:
            f.write(response.content)

        print(f"Model downloaded successfully to: {model_path}")
        return True
    except Exception as e:
        print(f"Failed to download model: {str(e)}")
        return False

def load_model(model_path="models/tomato_disease_model.h5"):
    try:
        if not os.path.exists(model_path):
            hf_url = "https://huggingface.co/tveesha15/tomato-disease-model/resolve/main/tomato_disease_model.h5"
            success = download_model_from_huggingface(hf_url, model_path)
            if not success or not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path} after download.")

        print(f"Loading model from: {model_path}")
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")

        if not isinstance(model, tf.keras.Model):
            raise ValueError("Loaded object is not a valid Keras model")

        if model.output_shape[-1] != len(DISEASE_CLASSES):
            raise ValueError(f"Model output shape {model.output_shape[-1]} does not match number of classes {len(DISEASE_CLASSES)}")

        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None
