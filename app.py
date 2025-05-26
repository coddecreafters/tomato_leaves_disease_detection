from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from model import load_model, predict_disease, DISEASE_CLASSES
import tensorflow as tf
import gc
import logging
import tempfile
import shutil
import requests
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Set TensorFlow to use CPU only and reduce logging
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configure TensorFlow memory settings
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Limit TensorFlow memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        logger.warning(f"GPU memory growth setting failed: {e}")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(tempfile.gettempdir(), 'tomato_disease_uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Model configuration
MODEL_DIR = Path('models')
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / 'tomato_disease_model.h5'
MODEL_URL = "https://huggingface.co/tveesha15/tomato-disease-model/resolve/main/tomato_disease_model.h5"

# Initialize model as None
model = None

def download_model():
    """Download the model from Hugging Face if it doesn't exist locally."""
    try:
        logger.info(f"Downloading model from {MODEL_URL}")
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        
        with open(MODEL_PATH, 'wb') as file:
            for data in response.iter_content(block_size):
                file.write(data)
                
        logger.info("Model downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        return False

def load_model_safely():
    """Load the model safely, downloading it if necessary."""
    global model
    try:
        if model is None:
            logger.info("Starting model load process...")
            logger.info(f"Model path: {MODEL_PATH}")
            
            # Check if model file exists, download if it doesn't
            if not MODEL_PATH.exists():
                logger.info("Model file not found locally, attempting to download...")
                if not download_model():
                    logger.error("Failed to download model")
                    return False
            
            logger.info("Loading model from file...")
            
            # Clear memory before loading
            gc.collect()
            
            # Load model with memory optimization
            with tf.device('/CPU:0'):
                model = load_model(str(MODEL_PATH))
            
            if model is None:
                logger.error("Model loaded but returned None")
                return False
                
            logger.info("Model loaded successfully")
            logger.info(f"Model input shape: {model.input_shape}")
            logger.info(f"Model output shape: {model.output_shape}")
            
            # Clear any unused memory
            gc.collect()
            logger.info("Garbage collection completed")
            
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

@app.before_first_request
def initialize():
    """Initialize the application and load the model."""
    logger.info("Initializing application...")
    if not load_model_safely():
        logger.error("Failed to initialize model")
    else:
        logger.info("Application initialized successfully")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    global model
    try:
        if 'file' not in request.files:
            logger.warning("No file uploaded")
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['file']
        if file.filename == '':
            logger.warning("No file selected")
            return jsonify({'error': 'No file selected'}), 400
            
        if file:
            # Create a unique filename
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Ensure the upload directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            logger.info(f"Saving file to {filepath}")
            file.save(filepath)
            
            # Verify file was saved
            if not os.path.exists(filepath):
                logger.error(f"File was not saved successfully at {filepath}")
                return jsonify({'error': 'Failed to save uploaded file'}), 500
            
            try:
                # Ensure model is loaded
                if not load_model_safely():
                    logger.error("Model not loaded properly")
                    return jsonify({'error': 'Model not loaded properly. Please check server logs.'}), 500
                
                # Make prediction
                logger.info("Making prediction...")
                with tf.device('/CPU:0'):
                    predictions = predict_disease(model, filepath)
                
                if predictions is None:
                    logger.error("Prediction failed")
                    return jsonify({'error': 'Failed to make prediction'}), 500
                    
                # Get top prediction
                top_prediction = DISEASE_CLASSES[predictions.argmax()]
                confidence = float(predictions.max())
                logger.info(f"Prediction: {top_prediction} with confidence {confidence}")
                
                return jsonify({
                    'prediction': top_prediction,
                    'confidence': confidence
                })
            finally:
                # Clean up uploaded file
                try:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                        logger.info("Cleaned up uploaded file")
                except Exception as e:
                    logger.warning(f"Failed to clean up file: {str(e)}")
                
                # Force garbage collection
                gc.collect()
                logger.info("Garbage collection completed after prediction")
            
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.teardown_appcontext
def cleanup(exception=None):
    """Clean up temporary files when the application context ends."""
    try:
        if os.path.exists(app.config['UPLOAD_FOLDER']):
            shutil.rmtree(app.config['UPLOAD_FOLDER'])
            logger.info("Cleaned up temporary directory")
    except Exception as e:
        logger.warning(f"Failed to clean up temporary directory: {str(e)}")

if __name__ == '__main__':
    # Ensure model is loaded before starting server
    logger.info("Starting application...")
    if not load_model_safely():
        logger.warning("Model not loaded properly. The application may not function correctly.")
    port = int(os.environ.get('PORT', 10000))
    logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port)
