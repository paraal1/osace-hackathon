"""
CNN Model architecture for Robot vs Human classification.
Uses transfer learning with MobileNetV2.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
try:
    from src import config
except ImportError:
    from . import config


def create_model(input_shape=(224, 224, 3), num_classes=2, use_pretrained=True):
    """
    Create a CNN model for image classification.
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of output classes
        use_pretrained: Whether to use transfer learning with MobileNetV2
    
    Returns:
        Compiled Keras model
    """
    print("\n" + "=" * 50)
    print("BUILDING MODEL")
    print("=" * 50)
    
    if use_pretrained:
        print("Using Transfer Learning with MobileNetV2")
        
        # Load pre-trained MobileNetV2 without top layers
        base_model = MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze the base model layers
        base_model.trainable = False
        
        # Create new model on top
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
    else:
        print("Building custom CNN from scratch")
        
        # Custom CNN architecture
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and Dense Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    print("\nModel architecture:")
    model.summary()
    
    return model


def unfreeze_model(model, num_layers_to_unfreeze=20):
    """
    Unfreeze the last few layers of the base model for fine-tuning.
    
    Args:
        model: Keras model
        num_layers_to_unfreeze: Number of layers to unfreeze from the end
    
    Returns:
        Model with unfrozen layers
    """
    # Get the base model (first layer)
    base_model = model.layers[0]
    
    # Unfreeze the last num_layers_to_unfreeze layers
    base_model.trainable = True
    
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False
    
    # Recompile with a lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    print(f"\nUnfroze last {num_layers_to_unfreeze} layers for fine-tuning")
    
    return model


if __name__ == "__main__":
    # Test model creation
    model = create_model(
        input_shape=(*config.IMG_SIZE, 3),
        num_classes=config.NUM_CLASSES,
        use_pretrained=True
    )
    
    print(f"\nTotal parameters: {model.count_params():,}")
    print(f"Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")