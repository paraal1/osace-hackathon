# ?? Quick Start Guide

Get up and running with the Robot vs Human Classifier in minutes!

## ? Super Quick Start (For Testing)

If you just want to test the project with the existing test images:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create sample dataset from test images
python download_dataset.py
# Choose option 3

# 3. Run the interactive setup
python setup.py
# Follow the menu options
```

## ?? Full Setup (For Production)

### Step 1: Environment Setup

```bash
# Clone the repository
git clone https://github.com/paraal1/osace-hackathon.git
cd osace-hackathon

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Prepare Dataset

**Option A: Use existing dataset**
```bash
# Organize your images:
# data/raw/human/ - put human images here
# data/raw/robot/ - put robot images here

# Then run preprocessing
python src/data_preprocessing.py
```

**Option B: Download from Kaggle**
```bash
# Install Kaggle API
pip install kaggle

# Setup Kaggle credentials
# Download kaggle.json from https://www.kaggle.com/settings
# Place in ~/.kaggle/ (Linux/Mac) or C:\Users\<user>\.kaggle\ (Windows)

# Run download helper
python download_dataset.py
```

**Option C: Create sample dataset** (for testing only)
```bash
python download_dataset.py
# Choose option 3
```

### Step 3: Train the Model

```bash
python src/train.py
```

This will:
- ? Load and augment training data
- ? Create MobileNetV2-based model
- ? Train for 10 epochs
- ? Save best model to `models/robot_vs_human_classifier.h5`
- ? Generate training visualizations
- ? Create performance report

**Expected output:**
```
Training samples: 1400
Validation samples: 300
Epoch 1/10: loss: 0.3421 - accuracy: 0.8571 - val_loss: 0.2156 - val_accuracy: 0.9133
...
Epoch 10/10: loss: 0.0832 - accuracy: 0.9657 - val_loss: 0.1243 - val_accuracy: 0.9533

Model saved to: models/robot_vs_human_classifier.h5
```

### Step 4: Test Predictions

**CLI Testing:**
```bash
python src/predict.py
```

**Web Interface:**
```bash
python api/app.py
# Open browser: http://localhost:5000
```

## ?? Common Workflows

### Workflow 1: Quick Demo

```bash
# 1. Install
pip install -r requirements.txt

# 2. Use sample data
python download_dataset.py  # Option 3

# 3. Train (quick - just a few images)
python src/train.py

# 4. Test
python api/app.py
```

### Workflow 2: Full Training

```bash
# 1. Setup environment
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# 2. Get real dataset (100+ images per class)
# Organize in data/raw/human/ and data/raw/robot/

# 3. Preprocess
python src/data_preprocessing.py

# 4. Train
python src/train.py

# 5. Evaluate
python src/predict.py

# 6. Deploy
python api/app.py
```

### Workflow 3: Interactive Menu

```bash
python setup.py
```

Follow the interactive menu for guided setup.

## ?? Configuration

Edit `src/config.py` to customize:

```python
# Model parameters
IMG_SIZE = (224, 224)  # Input image size
BATCH_SIZE = 32        # Training batch size
EPOCHS = 10     # Number of training epochs
LEARNING_RATE = 0.0001 # Learning rate

# Data split
TRAIN_RATIO = 0.7   # 70% for training
VAL_RATIO = 0.15   # 15% for validation
TEST_RATIO = 0.15    # 15% for testing
```

## ?? Troubleshooting

### Issue: "Module not found" errors

**Solution:**
```bash
# Make sure you're in the project root directory
cd osace-hackathon

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: "Model file not found"

**Solution:**
```bash
# Train the model first
python src/train.py
```

### Issue: RAW images not loading

**Solution:**
```bash
# Install rawpy
pip install rawpy

# If still fails, convert RAW to JPG manually
```

### Issue: Out of memory during training

**Solution:**
Edit `src/config.py`:
```python
BATCH_SIZE = 16  # Reduce from 32
IMG_SIZE = (128, 128)  # Reduce from (224, 224)
```

### Issue: Low accuracy

**Solutions:**
- Use more training data (100+ images per class minimum)
- Enable fine-tuning in `src/train.py`
- Increase number of epochs
- Check data quality and variety

## ?? Expected Performance

With a good dataset (500+ images per class):

| Metric | Expected Value |
|--------|---------------|
| Training Accuracy | 95-98% |
| Validation Accuracy | 90-95% |
| Training Time (CPU) | 30-60 min |
| Training Time (GPU) | 5-10 min |
| Prediction Time | < 1 second |

## ?? Next Steps

After successful setup:

1. **Explore the data**: Open `notebooks/data_exploration.ipynb`
2. **Tune the model**: Adjust hyperparameters in `src/config.py`
3. **Test thoroughly**: Use the web interface to test various images
4. **Deploy**: Consider deploying to cloud (Heroku, AWS, Azure)
5. **Improve**: Add more features (batch prediction, model versioning)

## ?? Additional Resources

- [TensorFlow Documentation](https://www.tensorflow.org/guide)
- [Keras Documentation](https://keras.io/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)

## ?? Tips

- **Start small**: Test with a few images first
- **Use GPU**: Much faster training if available
- **Monitor training**: Watch for overfitting (val_loss increasing)
- **Save checkpoints**: Best model is auto-saved during training
- **Validate often**: Use the web interface to test real-world images

## ? Need Help?

- Check the [README.md](README.md) for detailed documentation
- Open an issue on GitHub
- Review the code comments in each module

---

Happy classifying! ????
