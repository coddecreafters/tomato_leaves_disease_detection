from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from model import load_model, predict_disease, DISEASE_CLASSES
import tensorflow as tf

# Configure TensorFlow memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Error setting memory growth: {e}")

# Set TensorFlow to use CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model at startup
model_path = os.path.join('models', 'tomato_disease_model.h5')
model = None

@app.before_first_request
def initialize():
    global model
    try:
        model = load_model(model_path)
        if model is None:
            print("Failed to load model during initialization")
    except Exception as e:
        print(f"Error during model initialization: {str(e)}")

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
            global model
            if model is None:
                print("Model not loaded, attempting to load...")
                model = load_model(model_path)
                if model is None:
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
                
            return jsonify({
                'prediction': top_prediction,
                'confidence': confidence
            })
            
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Ensure model is loaded before starting server
    model = load_model(model_path)
    if model is None:
        print("Warning: Model not loaded properly. The application may not function correctly.")
    app.run(host='0.0.0.0', port=10000)
