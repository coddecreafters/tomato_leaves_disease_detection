import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import create_model, DISEASE_CLASSES
import os

# Constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 5

def train_model():
    # Create data generators with augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Create data generator for validation (only rescaling)
    val_datagen = ImageDataGenerator(rescale=1./255)

    # Load and preprocess the training data
    train_generator = train_datagen.flow_from_directory(
        'tomato_dataset/train',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    validation_generator = val_datagen.flow_from_directory(
        'tomato_dataset/val',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    # Create and compile the model
    model = create_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=len(DISEASE_CLASSES))
    
    # Add callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.2,
            patience=2
        )
    ]

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        callbacks=callbacks
    )

    return model, history

if __name__ == '__main__':
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    print("Starting model training...")
    model, history = train_model()
    print("Training completed!")
    print("Model saved to models/tomato_disease_model.h5")
    model.save('models/tomato_disease_model.h5')
    print("Model saved successfully!") 