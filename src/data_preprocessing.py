"""
Data preprocessing and augmentation for the image classifier.
Handles loading, splitting, and preparing the dataset.
"""
import os
import shutil
import numpy as np
import cv2
import rawpy
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Allow PIL to load truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Robust import of config whether run as module or script
try:
    from src import config  # when imported from project root
except ImportError:
    try:
        from . import config  # fallback using relative import
    except ImportError:
        import config  # last resort for direct execution


def load_and_convert_image(image_path, target_size=(224, 224)):
    """
    Load an image and convert it to RGB format.
    Handles various formats including RAW files.
    
    Args:
        image_path: Path to the image file
        target_size: Tuple of (height, width) for resizing
    
    Returns:
        numpy array of the processed image
    """
    file_ext = os.path.splitext(image_path)[1].lower()
    
    try:
        # Handle RAW formats (ARW, CR2, NEF, etc.)
        if file_ext in ['.arw', '.cr2', '.nef', '.raw', '.dng']:
            with rawpy.imread(image_path) as raw:
                rgb = raw.postprocess()
                img = Image.fromarray(rgb)
        else:
            # Handle standard formats
            img = Image.open(image_path).convert('RGB')
        
        # Resize image
        img = img.resize(target_size, Image.LANCZOS)
        return np.array(img)
    
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def download_and_prepare_dataset():
    """
    Download dataset from Kaggle or prepare existing dataset.
    This is a placeholder - you'll need to add your dataset source.
    """
    print("=" * 50)
    print("DATASET PREPARATION")
    print("=" * 50)
    
    # For now, we'll use a placeholder message
    print("\nTo prepare the dataset, please:")
    print("1. Download a 'Robots vs Humans' dataset from Kaggle or similar")
    print("2. Extract it to the 'data/raw' directory")
    print("3. Organize images in subdirectories: data/raw/human/ and data/raw/robot/")
    print("\nAlternatively, you can use a pre-existing dataset.")
    print("\nExample Kaggle datasets:")
    print("- Human vs Robot Images")
    print("- Person vs Robot Classification")
    
    # Check if raw data exists
    if not os.path.exists(config.RAW_DATA_DIR):
        os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    
    return True


def split_dataset(source_dir, train_dir, val_dir, test_dir, 
            train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        source_dir: Directory containing raw data with class subdirectories
        train_dir: Output directory for training data
        val_dir: Output directory for validation data
        test_dir: Output directory for test data
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
    """
    print("\n" + "=" * 50)
    print("SPLITTING DATASET")
    print("=" * 50)
    
    # Create directories if they don't exist
    for directory in [train_dir, val_dir, test_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Process each class
    for class_name in config.CLASS_LABELS:
        class_source = os.path.join(source_dir, class_name)
        
        if not os.path.exists(class_source):
            print(f"Warning: Class directory not found: {class_source}")
            continue
        
        # Get all image files
        image_files = [f for f in os.listdir(class_source) 
                        if os.path.isfile(os.path.join(class_source, f))]
        
        if len(image_files) == 0:
            print(f"Warning: No images found for class '{class_name}'")
            continue
        
        print(f"\nProcessing class '{class_name}': {len(image_files)} images")
        
        # Split into train and temp (val + test)
        train_files, temp_files = train_test_split(
            image_files, 
            test_size=(val_ratio + test_ratio),
            random_state=42
        )
        
        # Split temp into val and test
        val_files, test_files = train_test_split(
            temp_files,
            test_size=test_ratio/(val_ratio + test_ratio),
            random_state=42
        )
        
        # Copy files to respective directories with per-split logging
        for split_name, files, target_dir in [
            ('train', train_files, train_dir),
            ('val', val_files, val_dir),
            ('test', test_files, test_dir)
        ]:
            class_target = os.path.join(target_dir, class_name)
            os.makedirs(class_target, exist_ok=True)
            
            for file in files:
                src = os.path.join(class_source, file)
                dst = os.path.join(class_target, file)
                shutil.copy2(src, dst)
            print(f"  {split_name}: {len(files)} images")
    
    print("\nDataset split completed!")


def create_data_generators(train_dir, val_dir, batch_size=32, img_size=(224, 224)):
    """
    Create data generators with augmentation for training and validation.
    
    Args:
        train_dir: Directory containing training data
        val_dir: Directory containing validation data
        batch_size: Batch size for training
        img_size: Target image size
     
    Returns:
        Tuple of (train_generator, val_generator)
    """
    print("\n" + "=" * 50)
    print("CREATING DATA GENERATORS")
    print("=" * 50)
    
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest'
    )
    
    # Validation data generator (no augmentation, only rescaling)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {val_generator.samples}")
    print(f"Classes: {train_generator.class_indices}")

    return train_generator, val_generator


if __name__ == "__main__":
    # Download/prepare dataset
    download_and_prepare_dataset()
    
    # Split dataset
    if os.path.exists(config.RAW_DATA_DIR):
        split_dataset(
            config.RAW_DATA_DIR,
            config.TRAIN_DIR,
            config.VAL_DIR,
            config.TEST_DIR,
            config.TRAIN_RATIO,
            config.VAL_RATIO,
            config.TEST_RATIO
        )
    
    # Create data generators
    if os.path.exists(config.TRAIN_DIR) and os.path.exists(config.VAL_DIR):
        train_gen, val_gen = create_data_generators(
            config.TRAIN_DIR,
            config.VAL_DIR,
            config.BATCH_SIZE,
            config.IMG_SIZE
        )
        print("\nData preprocessing completed successfully!")
    else:
        print("\nPlease prepare the dataset first!")