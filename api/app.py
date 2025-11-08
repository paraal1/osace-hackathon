"""
Flask API for Robot vs Human Image Classifier.
Provides endpoints for image upload and prediction.
"""
import os
import sys
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from datetime import datetime
import numpy as np

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.predict import load_model, predict_image
from api.database import init_database, save_prediction, get_all_predictions, get_statistics

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = config.MAX_FILE_SIZE
app.config['UPLOAD_FOLDER'] = config.UPLOAD_DIR

# Global model variable
model = None
initialized = False

def initialize_app():
    """Initialize model and database."""
    global model, initialized
    
    if initialized:
        return
    
    # Create upload directory
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    
    # Initialize database
    init_database()
    
    # Load model
    try:
        print("Loading model...")
        model = load_model()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please train the model first!")
    
    initialized = True


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Render the main page."""
    initialize_app()
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    initialize_app()
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict the class of an uploaded image.
    
    Expected: multipart/form-data with 'file' field
    Returns: JSON with prediction results
    """
    global model
    
    initialize_app()
    
    # Check if model is loaded
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please train the model first.'
        }), 500
 
    # Check if file is present
    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No file provided'
        }), 400
    
    file = request.files['file']
    
    # Check if file is selected
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No file selected'
        }), 400
    
    # Check if file type is allowed
    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': f'File type not allowed. Allowed types: {", ".join(config.ALLOWED_EXTENSIONS)}'
        }), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(config.UPLOAD_DIR, unique_filename)
        file.save(filepath)
        
        # Make prediction
        predicted_class, confidence, probabilities = predict_image(model, filepath)
     
        if predicted_class is None:
            return jsonify({
                'success': False,
                'error': 'Failed to process image'
            }), 500
        
        # Convert probabilities to dict
        prob_dict = {
            class_name: float(prob) 
            for class_name, prob in zip(config.CLASS_LABELS, probabilities)
        }
        
        # Save to database
        save_prediction(
            filename=unique_filename,
            predicted_class=predicted_class,
            confidence=float(confidence),
            image_path=filepath,
            probabilities=prob_dict
        )
        
        # Return results
        return jsonify({
            'success': True,
            'filename': unique_filename,
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'probabilities': prob_dict,
            'timestamp': datetime.now().isoformat()
        })
  
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/history', methods=['GET'])
def history():
    """Get prediction history."""
    initialize_app()
    try:
        limit = request.args.get('limit', default=10, type=int)
        predictions = get_all_predictions(limit=limit)
    
        # Format results
        results = []
        for pred in predictions:
            results.append({
                'id': pred[0],
                'filename': pred[1],
                'predicted_class': pred[2],
                'confidence': pred[3],
                'timestamp': pred[4]
            })
        
        return jsonify({
            'success': True,
            'predictions': results,
            'count': len(results)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/statistics', methods=['GET'])
def statistics():
    """Get prediction statistics."""
    initialize_app()
    try:
        stats = get_statistics()
        
        return jsonify({
            'success': True,
            'statistics': stats
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    return send_from_directory(config.UPLOAD_DIR, filename)


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error."""
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 16MB.'
    }), 413


if __name__ == '__main__':
    print("=" * 50)
    print("STARTING FLASK API SERVER")
    print("=" * 50)
    print(f"\nServer will be available at: http://localhost:5000")
    print(f"API endpoints:")
    print(f"  - GET  /api/health        - Health check")
    print(f"  - POST /api/predict       - Upload and predict image")
    print(f"  - GET  /api/history       - Get prediction history")
    print(f"  - GET  /api/statistics    - Get statistics")
    print()
    
    # Initialize before running
    initialize_app()
    
    app.run(debug=True, host='0.0.0.0', port=5000)