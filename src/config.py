"""
Configuration file for the Image Classifier project.
"""
import os

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')

# Data directories
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# Model parameters
IMG_SIZE = (224, 224)  # Standard size for transfer learning models
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0001

# Model architecture
MODEL_NAME = 'robot_vs_human_classifier.h5'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# Class labels
CLASS_LABELS = ['human', 'robot']
NUM_CLASSES = len(CLASS_LABELS)

# Data split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Database
DB_PATH = os.path.join(BASE_DIR, 'predictions.db')

# API settings
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'arw', 'raw', 'tif', 'tiff'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16 MB
