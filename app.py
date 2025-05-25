from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from model import load_model, predict_disease, DISEASE_CLASSES
import tensorflow as tf
import gc
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Set TensorFlow to use CPU only and reduce logging
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configure TensorFlow memory settings
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
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
            logger.info("Loading model...")
            model = load_model(model_path)
            if model is None:
                logger.error("Failed to load model")
                return False
            logger.info("Model loaded successfully")
            # Clear any unused memory
            gc.collect()
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

@app.before_first_request
def initialize():
    load_model_safely()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Ensure model is loaded
            if not load_model_safely():
                return jsonify({'error': 'Model not loaded properly. Please check server logs.'}), 500
            
            # Make prediction
            predictions = predict_disease(model, filepath)
            if predictions is None:
                return jsonify({'error': 'Failed to make prediction'}), 500
                
            # Get top prediction
            top_prediction = DISEASE_CLASSES[predictions.argmax()]
            confidence = float(predictions.max())
            
            # Clean up uploaded file
            try:
                os.remove(filepath)
            except:
                pass
                
            # Force garbage collection
            gc.collect()
                
            return jsonify({
                'prediction': top_prediction,
                'confidence': confidence
            })
            
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Ensure model is loaded before starting server
    logger.info("Starting application...")
    if not load_model_safely():
        logger.warning("Model not loaded properly. The application may not function correctly.")
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
