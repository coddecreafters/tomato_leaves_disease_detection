from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from model import load_model, predict_disease, DISEASE_CLASSES
import tensorflow as tf
import gc
import logging
import tempfile
import shutil

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

# Load model at startup
model_path = os.path.join('models', 'tomato_disease_model.h5')
model = None

def load_model_safely():
    global model
    try:
        if model is None:
            logger.info("Starting model load process...")
            logger.info(f"Model path: {model_path}")
            
            # Check if model file exists
            if not os.path.exists(model_path):
                logger.error(f"Model file not found at {model_path}")
                return False
                
            logger.info("Loading model from file...")
            model = load_model(model_path)
            
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
