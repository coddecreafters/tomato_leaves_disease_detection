from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from model import load_model, predict_disease, DISEASE_CLASSES
import numpy as np

# Get the absolute path to the project directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__, 
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)

# Configure upload folder
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model with error handling
try:
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'tomato_disease_model.h5')
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    if model is None:
        raise ValueError("Model failed to load")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded properly. Please check server logs.'})
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Make prediction
            predictions = predict_disease(model, filepath)
            if predictions is None:
                return jsonify({'error': 'Failed to make prediction'})
                
            predicted_class = DISEASE_CLASSES[np.argmax(predictions)]
            confidence = float(np.max(predictions))
            
            # Clean up
            os.remove(filepath)
            
            return jsonify({
                'disease': predicted_class,
                'confidence': confidence
            })
        except Exception as e:
            # Clean up file if it exists
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Prediction error: {str(e)}'})
    
    return jsonify({'error': 'Invalid file type'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))  # Changed to 10000 to match Render's default
    app.run(host='0.0.0.0', port=port)
