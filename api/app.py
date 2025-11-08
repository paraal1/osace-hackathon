"""
Flask API for Robot vs Human Image Classifier.
Provides endpoints for image upload and prediction.
"""
import os
import sys
from flask import Flask, request, jsonify, render_template, send_from_directory, make_response
from werkzeug.utils import secure_filename
from datetime import datetime
import numpy as np
import csv
from io import StringIO
import cv2

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

# Store last batch results for CSV export
last_batch_results = None

# Initialize face detection for camera distance feature
face_cascade = None

def initialize_app():
    """Initialize model and database."""
    global model, initialized, face_cascade
    
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
    
    # Load face cascade for camera distance detection
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        print("Face detection loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load face detection: {e}")
    
    initialized = True


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS


def calculate_distance_status(face_width, frame_width):
    """
    Determine if user is at correct distance based on face size.
    
    Args:
        face_width: Width of detected face in pixels
        frame_width: Width of video frame in pixels
    
    Returns:
        tuple: (status_color, face_ratio, message)
            - status_color: 'green', 'yellow', or 'red'
            - face_ratio: percentage of frame width occupied by face
            - message: guidance message for user
    """
    # Face should be 25-35% of frame width for optimal distance
    face_ratio = (face_width / frame_width) * 100
    
    if 25 <= face_ratio <= 35:
        return 'green', face_ratio, 'Perfect distance! Stay there.'
    elif 20 <= face_ratio < 25:
        return 'yellow', face_ratio, 'Move a bit closer'
    elif 35 < face_ratio <= 40:
        return 'yellow', face_ratio, 'Move a bit back'
    elif face_ratio < 20:
        return 'red', face_ratio, 'Too far away - move closer'
    else:
        return 'red', face_ratio, 'Too close - move back'


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
        'face_detection_loaded': face_cascade is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/camera/distance_check', methods=['POST'])
def check_camera_distance():
    """
    Check if user is at correct distance from camera.
    
    Expected: multipart/form-data with 'frame' field (image file)
    Returns: JSON with distance feedback
    """
    global face_cascade
    
    initialize_app()
    
    # Check if face detection is loaded
    if face_cascade is None:
        return jsonify({
            'success': False,
            'error': 'Face detection not available'
        }), 500
    
    # Check if frame is present
    if 'frame' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No frame provided'
        }), 400
    
    file = request.files['frame']
    
    try:
        # Read image from upload
        img_bytes = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({
                'success': False,
                'error': 'Failed to decode image'
            }), 400
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return jsonify({
                'success': True,
                'status': 'red',
                'message': 'No face detected - please face the camera',
                'distance_ok': False,
                'face_detected': False
            })
        
        # Use largest face (closest to camera)
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        
        # Calculate distance status
        status, ratio, message = calculate_distance_status(w, frame.shape[1])
        
        return jsonify({
            'success': True,
            'status': status,
            'face_ratio': round(ratio, 2),
            'distance_ok': status == 'green',
            'face_detected': True,
            'message': message,
            'bbox': {
                'x': int(x), 
                'y': int(y), 
                'w': int(w), 
                'h': int(h)
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error processing frame: {str(e)}'
        }), 500


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


@app.route('/api/predict/batch', methods=['POST'])
def batch_predict():
    """
    Predict the class of multiple uploaded images.
    
    Expected: multipart/form-data with 'files' field (multiple files)
    Returns: JSON with batch prediction results and summary
    """
    global model, last_batch_results
    
    initialize_app()
    
    # Check if model is loaded
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please train the model first.'
        }), 500
    
    # Check if files are present
    if 'files' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No files provided'
        }), 400
    
    files = request.files.getlist('files')
    
    # Check if any files were selected
    if len(files) == 0 or (len(files) == 1 and files[0].filename == ''):
        return jsonify({
            'success': False,
            'error': 'No files selected'
        }), 400
    
    # Limit batch size
    MAX_BATCH_SIZE = 20
    if len(files) > MAX_BATCH_SIZE:
        return jsonify({
            'success': False,
            'error': f'Too many files. Maximum batch size is {MAX_BATCH_SIZE}.'
        }), 400
    
    results = []
    errors = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    batch_timestamp = datetime.now().isoformat()
    
    for idx, file in enumerate(files):
        try:
            # Check if file type is allowed
            if not allowed_file(file.filename):
                errors.append({
                    'filename': file.filename,
                    'error': f'File type not allowed. Allowed types: {", ".join(config.ALLOWED_EXTENSIONS)}'
                })
                continue
            
            # Save uploaded file
            filename = secure_filename(file.filename)
            unique_filename = f"{timestamp}_{idx}_{filename}"
            filepath = os.path.join(config.UPLOAD_DIR, unique_filename)
            file.save(filepath)
            
            # Make prediction
            predicted_class, confidence, probabilities = predict_image(model, filepath)
            
            if predicted_class is None:
                errors.append({
                    'filename': filename,
                    'error': 'Failed to process image'
                })
                continue
            
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
            
            # Add to results
            results.append({
                'filename': unique_filename,
                'original_filename': filename,
                'predicted_class': predicted_class,
                'confidence': float(confidence),
                'probabilities': prob_dict,
                'timestamp': batch_timestamp
            })
        
        except Exception as e:
            errors.append({
                'filename': file.filename,
                'error': str(e)
            })
    
    # Calculate summary statistics
    robot_count = sum(1 for r in results if r['predicted_class'] == 'robot')
    human_count = sum(1 for r in results if r['predicted_class'] == 'human')
    avg_confidence = sum(r['confidence'] for r in results) / len(results) if results else 0
    
    # Store results for CSV export
    last_batch_results = {
        'results': results,
        'timestamp': batch_timestamp,
        'summary': {
            'total': len(files),
            'processed': len(results),
            'failed': len(errors),
            'robot_count': robot_count,
            'human_count': human_count,
            'average_confidence': avg_confidence
        }
    }
    
    return jsonify({
        'success': True,
        'processed': len(results),
        'failed': len(errors),
        'total': len(files),
        'results': results,
        'errors': errors if errors else None,
        'summary': {
            'robot_count': robot_count,
            'human_count': human_count,
            'average_confidence': float(avg_confidence),
            'high_confidence_predictions': sum(1 for r in results if r['confidence'] > 0.9),
            'low_confidence_predictions': sum(1 for r in results if r['confidence'] < 0.7)
        },
        'timestamp': batch_timestamp
    })


@app.route('/api/export/batch/csv', methods=['GET'])
def export_batch_csv():
    """
    Export the last batch prediction results to CSV.
    
    Returns: CSV file download
    """
    global last_batch_results
    
    if last_batch_results is None or not last_batch_results.get('results'):
        return jsonify({
            'success': False,
            'error': 'No batch results available to export. Please run a batch prediction first.'
        }), 404
    
    try:
        # Create CSV in memory
        output = StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            'Filename',
            'Original Filename',
            'Predicted Class',
            'Confidence (%)',
            'Robot Probability (%)',
            'Human Probability (%)',
            'Timestamp'
        ])
        
        # Write data rows
        for result in last_batch_results['results']:
            writer.writerow([
                result['filename'],
                result['original_filename'],
                result['predicted_class'].upper(),
                f"{result['confidence'] * 100:.2f}",
                f"{result['probabilities']['robot'] * 100:.2f}",
                f"{result['probabilities']['human'] * 100:.2f}",
                result['timestamp']
            ])
        
        # Add summary section
        writer.writerow([])
        writer.writerow(['BATCH SUMMARY'])
        writer.writerow(['Total Files', last_batch_results['summary']['total']])
        writer.writerow(['Successfully Processed', last_batch_results['summary']['processed']])
        writer.writerow(['Failed', last_batch_results['summary']['failed']])
        writer.writerow(['Robot Count', last_batch_results['summary']['robot_count']])
        writer.writerow(['Human Count', last_batch_results['summary']['human_count']])
        writer.writerow(['Average Confidence (%)', f"{last_batch_results['summary']['average_confidence'] * 100:.2f}"])
        
        # Create response
        output.seek(0)
        response = make_response(output.getvalue())
        response.headers['Content-Type'] = 'text/csv'
        response.headers['Content-Disposition'] = f'attachment; filename=batch_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        
        return response
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to generate CSV: {str(e)}'
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
    print(f"  - GET  /api/health                  - Health check")
    print(f"  - POST /api/predict                 - Upload and predict image")
    print(f"  - POST /api/predict/batch           - Upload and predict multiple images")
    print(f"  - POST /api/camera/distance_check   - Check camera distance (NEW)")
    print(f"  - GET  /api/export/batch/csv        - Export last batch results to CSV")
    print(f"  - GET  /api/history                 - Get prediction history")
    print(f"  - GET  /api/statistics              - Get statistics")
    print()

    # Initialize before running
    initialize_app()
    
    app.run(debug=True, host='0.0.0.0', port=5000)