<img width="1907" height="875" alt="Screenshot 2025-11-08 201213" src="https://github.com/user-attachments/assets/58f8405b-361d-4e3a-9bcb-9eadacac36d1" />﻿# 🤖 AI Vision System - Robot vs Human Classifier

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A privacy-first, real-time web application that classifies images as "Human" or "Robot" using deep learning with MobileNetV2 architecture.

---

## 📸 Screenshots

<img width="1907" height="875" alt="Screenshot 2025-11-08 201213" src="https://github.com/user-attachments/assets/d6ef749f-0c1a-4adf-b68c-8ce69e765f09" />


### 🎯 Single Image Prediction & Results
![Prediction Results](imagesreadme/Screenshot%202025-11-08%20201304.png)
*Detailed confidence scores and probability breakdown for each class*
<img width="1242" height="870" alt="Screenshot 2025-11-08 201304" src="https://github.com/user-attachments/assets/5cc70fbb-9ce5-4084-b8aa-6bab831338e1" />

### 📊 Batch Processing Results
![Batch Results](imagesreadme/Screenshot%202025-11-08%20201342.png)
*Interactive grid view with per-image predictions and summary statistics*
<img width="1718" height="855" alt="Screenshot 2025-11-08 201342" src="https://github.com/user-attachments/assets/69b6ffb0-20ba-4cfc-95dc-40025087bc42" />

### 📈 Statistics & History Dashboard
![Statistics](imagesreadme/Screenshot%202025-11-08%20201430.png)
*Track total predictions, average confidence, and recent classification history*
<img width="815" height="834" alt="Screenshot 2025-11-08 201430" src="https://github.com/user-attachments/assets/315ed1ef-0a11-48b4-b52e-0296df908569" />

---

## 🚀 Features

- ✅ **Single & Batch Upload**: Analyze 1-20 images with drag-and-drop support
- ✅ **Live Camera Mode**: Real-time face detection with optimal distance guidance
- ✅ **High Accuracy**: 92-95% validation accuracy using MobileNetV2 transfer learning
- ✅ **Privacy-First**: All processing happens locally—no cloud uploads
- ✅ **Export Results**: Download predictions as CSV for analysis
- ✅ **Real-Time Feedback**: Animated UI with cyberpunk aesthetics

---

## 🎬 Quick Demo

> **Note**: Run the app locally to see it in action:

### 1. Clone the repository

### 2. Install dependencies

### 3. Run the Flask app

### 4. Access the app

Open your browser and go to `http://localhost:5000`

> **Tip**: Use sample images from the `Images/` folder or your own images in the supported formats (JPG, PNG, ARW, CR2, NEF, DNG).

## 📚 Documentation

See the [QUICKSTART.md](QUICKSTART.md) file for detailed documentation on:
- Project architecture
- Installation and configuration
- How to use the app
- API documentation
- Technical details on model architecture and training
- Troubleshooting tips and common issues

## ❓ FAQ & Troubleshooting

**Q: What to do if I encounter an error while uploading images?**
- Check the browser console (F12) for detailed error messages
- Ensure the image is in a supported format and within the size limits
- For CORS issues, ensure you're accessing the app from the correct origin (http://localhost:5000)

**Q: How to improve the prediction accuracy?**
- Use high-quality images with clear features
- Ensure the images are well-lit and focused
- Consider fine-tuning the model with additional data

**Q: What if the model is not loading or taking too long to respond?**
- Check the server logs for any error messages
- Ensure your system meets the hardware requirements (preferably with a dedicated GPU)
- Try restarting the Flask server and clearing the browser cache

For more FAQs, troubleshooting tips, and detailed documentation, visit the [wiki page](https://github.com/paraal1/osace-hackathon/wiki) or check the **Documentation** section in the app.

---

<div align="center">

# Thank you for using our AI Vision System!

## For any inquiries or support, please contact us at [multiverse_team@gmail.com](mailto:multiverse_team@gmail.com)

</div>
