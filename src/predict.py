"""
Prediction script for the Robot vs Human classifier.
"""
import os
import numpy as np
import rawpy
from PIL import Image
from tensorflow import keras
from src import config


def load_image_for_prediction(image_path, target_size=(224, 224)):
    """
    Load and preprocess an image for prediction.
    
    Args:
        image_path: Path to the image file
        target_size: Target size for the image
    
    Returns:
        Preprocessed image array ready for prediction
    """
    file_ext = os.path.splitext(image_path)[1].lower()
    
    try:
        # Handle RAW formats
        if file_ext in ['.arw', '.cr2', '.nef', '.raw', '.dng']:
            with rawpy.imread(image_path) as raw:
                rgb = raw.postprocess()
            img = Image.fromarray(rgb)
        else:
            # Handle standard formats
            img = Image.open(image_path).convert('RGB')
        
        # Resize
        img = img.resize(target_size, Image.LANCZOS)
        
        # Convert to array and normalize
        img_array = np.array(img) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def predict_image(model, image_path, class_labels=None):
    """
    Predict the class of an image.
    
    Args:
        model: Trained Keras model
        image_path: Path to the image file
        class_labels: List of class label names
    
    Returns:
     Tuple of (predicted_class, confidence, probabilities)
    """
    if class_labels is None:
        class_labels = config.CLASS_LABELS 
 
    # Load and preprocess image
    img_array = load_image_for_prediction(image_path, config.IMG_SIZE)
    
    if img_array is None:
        return None, None, None
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    
    # Get predicted class and confidence
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = class_labels[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx]
    
    return predicted_class, confidence, predictions[0]


def predict_batch(model, image_paths, class_labels=None):
    """
    Predict classes for multiple images.
    
    Args:
        model: Trained Keras model
        image_paths: List of image file paths
        class_labels: List of class label names
    
    Returns:
        List of tuples (image_path, predicted_class, confidence)
    """
    results = []
    
    for image_path in image_paths:
        predicted_class, confidence, _ = predict_image(model, image_path, class_labels)
        
        if predicted_class is not None:
            results.append((image_path, predicted_class, confidence))
        else:
            results.append((image_path, "Error", 0.0))
    
    return results


def load_model(model_path=None):
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to the saved model file
    
    Returns:
        Loaded Keras model
    """
    if model_path is None:
        model_path = config.MODEL_PATH
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    print("Model loaded successfully!")
    
    return model


if __name__ == "__main__":
    # Test prediction with example images
    print("=" * 50)
    print("TESTING PREDICTION")
    print("=" * 50)
    
    # Load model
    try:
        model = load_model()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please train the model first by running: python src/train.py")
        exit(1)
    
    # Test with images from the Images directory
    test_images_dir = os.path.join(config.BASE_DIR, 'Images')
    
    if os.path.exists(test_images_dir):
        test_images = [
            os.path.join(test_images_dir, f) 
            for f in os.listdir(test_images_dir) 
            if os.path.isfile(os.path.join(test_images_dir, f))
        ]
    
        if test_images:
            print(f"\nFound {len(test_images)} test images\n")
         
            for img_path in test_images:
                predicted_class, confidence, probabilities = predict_image(model, img_path)
       
                if predicted_class is not None:
                    print(f"Image: {os.path.basename(img_path)}")
                    print(f"  Predicted: {predicted_class.upper()}")
                    print(f"  Confidence: {confidence*100:.2f}%")
                    print(f"  Probabilities: {dict(zip(config.CLASS_LABELS, probabilities))}")
                    print()
        else:
            print(f"No test images found in {test_images_dir}")
    else:
        print(f"Test images directory not found: {test_images_dir}")
