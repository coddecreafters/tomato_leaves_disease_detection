# Tomato Leaves Disease Detection

This project uses deep learning to detect diseases in tomato plant leaves. The model can identify various diseases such as:
- Early Blight
- Late Blight
- Leaf Mold
- Septoria Leaf Spot
- Spider Mites
- Target Spot
- Yellow Leaf Curl Virus
- Mosaic Virus
- Bacterial Spot
- Healthy

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to `http://localhost:5000`

## Project Structure
- `app.py`: Main Flask application
- `model.py`: CNN model architecture and training code
- `static/`: Static files (CSS, JavaScript)
- `templates/`: HTML templates
- `uploads/`: Directory for uploaded images
- `models/`: Directory for saved model weights

## Dataset
The model is trained on the PlantVillage dataset, which contains images of healthy and diseased tomato leaves. 