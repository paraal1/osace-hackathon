"""
Training script for the Robot vs Human classifier.
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from src import config
from src.model import create_model, unfreeze_model
from src.data_preprocessing import create_data_generators


def plot_training_history(history, save_path='training_history.png'):
    """
    Plot training and validation accuracy/loss.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Precision
    if 'precision' in history.history:
        axes[1, 0].plot(history.history['precision'], label='Train Precision')
        axes[1, 0].plot(history.history['val_precision'], label='Val Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

    # Recall
    if 'recall' in history.history:
        axes[1, 1].plot(history.history['recall'], label='Train Recall')
        axes[1, 1].plot(history.history['val_recall'], label='Val Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining history plot saved to: {save_path}")
    plt.close()


def train_model(epochs=10, batch_size=32, use_pretrained=True, fine_tune=False):
    """
    Train the image classifier model.
    """
    print("\n" + "=" * 50)
    print("TRAINING ROBOT VS HUMAN CLASSIFIER")
    print("=" * 50)

    # Create data generators
    train_generator, val_generator = create_data_generators(
        config.TRAIN_DIR,
        config.VAL_DIR,
        batch_size,
        config.IMG_SIZE
    )

    # Create model
    model = create_model(
        input_shape=(*config.IMG_SIZE, 3),
        num_classes=config.NUM_CLASSES,
        use_pretrained=use_pretrained
    )

    # Define callbacks
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            config.MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # Train the model
    print("\n" + "=" * 50)
    print("STARTING TRAINING")
    print("=" * 50)

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )

    # Plot training history
    plot_training_history(history, os.path.join(config.MODEL_DIR, 'training_history.png'))

    # Fine-tuning (optional)
    if fine_tune and use_pretrained:
        print("\n" + "=" * 50)
        print("FINE-TUNING MODEL")
        print("=" * 50)

        model = unfreeze_model(model, num_layers_to_unfreeze=20)

        fine_tune_epochs = 5
        history_fine = model.fit(
            train_generator,
            epochs=fine_tune_epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )

        plot_training_history(
            history_fine,
            os.path.join(config.MODEL_DIR, 'fine_tuning_history.png')
        )

    # Evaluate on validation set
    print("\n" + "=" * 50)
    print("FINAL EVALUATION")
    print("=" * 50)

    results = model.evaluate(val_generator)
    # depending on compile metrics order
    if len(results) == 4:
        val_loss, val_accuracy, val_precision, val_recall = results
    else:
        val_loss, val_accuracy = results
        val_precision = val_recall = 0

    print(f"\nValidation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    print(f"Validation Precision: {val_precision:.4f}")
    print(f"Validation Recall: {val_recall:.4f}")

    # Calculate F1 Score
    if val_precision + val_recall > 0:
        f1_score = 2 * (val_precision * val_recall) / (val_precision + val_recall)
        print(f"Validation F1-Score: {f1_score:.4f}")
    else:
        f1_score = 0

    # Save final model
    model.save(config.MODEL_PATH)
    print(f"\nModel saved to: {config.MODEL_PATH}")

    # Save training summary
    summary_path = os.path.join(config.MODEL_DIR, 'training_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("TRAINING SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {'Transfer Learning (MobileNetV2)' if use_pretrained else 'Custom CNN'}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Image Size: {config.IMG_SIZE}\n")
        f.write(f"Learning Rate: {config.LEARNING_RATE}\n\n")
        f.write(f"Training Samples: {train_generator.samples}\n")
        f.write(f"Validation Samples: {val_generator.samples}\n\n")
        f.write(f"Final Validation Loss: {val_loss:.4f}\n")
        f.write(f"Final Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)\n")
        f.write(f"Final Validation Precision: {val_precision:.4f}\n")
        f.write(f"Final Validation Recall: {val_recall:.4f}\n")
        if val_precision + val_recall > 0:
            f.write(f"Final Validation F1-Score: {f1_score:.4f}\n")

    print(f"Training summary saved to: {summary_path}")

    return model, history


if __name__ == "__main__":
    model, history = train_model(
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        use_pretrained=True,
        fine_tune=True
    )

    print("\n" + "=" * 50)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 50)
