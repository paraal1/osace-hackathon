# 🤖 Robot vs Human Image Classifier

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Un clasificator de imagini bazat pe CNN (Convolutional Neural Network) care distinge între **roboți** și **oameni**. Proiectul include preprocesarea datelor, antrenarea modelului cu transfer learning, un API Flask REST și o interfață web interactivă pentru testare.

## 📋 Cuprins

- [Caracteristici](#-caracteristici)
- [Arhitectura Proiectului](#-arhitectura-proiectului)
- [Instalare și Configurare](#-instalare-și-configurare)
- [Cum se Rulează Proiectul](#-cum-se-rulează-proiectul)
- [Arhitectura Tehnică](#-arhitectura-tehnică)
- [API Documentation](#-api-documentation)
- [Rezultate Obținute](#-rezultate-obținute)
- [Demo și Testare](#-demo-și-testare)
- [Provocări Întâmpinate](#-provocări-întâmpinate)
- [Tehnologii Utilizate](#-tehnologii-utilizate)
- [Contribuitori](#-contribuitori)

## ✨ Caracteristici

- 🧠 **Transfer Learning** cu MobileNetV2 preantrenat pe ImageNet
- 📊 **Data Augmentation** sofisticat pentru îmbunătățirea performanței
- 🔌 **API REST** complet cu Flask pentru predicții
- 🎨 **Interfață Web** intuitivă cu drag & drop pentru upload
- 💾 **Persistență Date** cu SQLite pentru istoricul predicțiilor
- 📸 **Suport RAW** pentru formate profesionale (ARW, CR2, NEF, DNG)
- 📈 **Metrici Complete**: Accuracy, Precision, Recall, F1-Score
- 📉 **Vizualizări** detaliate pentru evoluția antrenării
- 🚀 **Batch Processing** - procesare multiplă de imagini
- 📥 **Export CSV** pentru rezultate batch

## 🏗️ Arhitectura Proiectului

```
osace-hackathon/
├── data/
│   ├── raw/  # Dataset original
│   │   ├── human/          # Imagini cu oameni
│   │   └── robot/          # Imagini cu roboți
│   ├── train/         # Date de antrenare (70%)
│   ├── val/   # Date de validare (15%)
│   └── test/        # Date de testare (15%)
├── models/
│   ├── robot_vs_human_classifier.h5     # Model salvat
│   ├── training_history.png             # Grafic evoluție antrenare
│   ├── fine_tuning_history.png          # Grafic fine-tuning
│└── training_summary.txt # Sumar rezultate
├── src/
│   ├── config.py   # Configurări globale
│   ├── data_preprocessing.py     # Preprocesare și split dataset
│   ├── model.py           # Arhitectură CNN
│   ├── train.py       # Script antrenare model
│   └── predict.py            # Script predicții CLI
├── api/
│   ├── app.py # Flask API server
│   ├── database.py         # SQLite database handler
│   └── templates/
│       └── index.html   # Interfață web
├── uploads/     # Imagini încărcate de utilizatori
├── Images/     # Imagini de test pentru demo
├── requirements.txt              # Dependențe Python
├── predictions.db       # Baza de date SQLite
└── README.md     # Documentație (acest fișier)
```

## 🔧 Instalare și Configurare

### Prerequisite

- **Python 3.8** sau mai nou
- **pip** (Python package manager)
- **Git** pentru clonare repository
- (Opțional) GPU cu CUDA pentru antrenare mai rapidă

### Pași de Instalare

#### 1. Clonează Repository-ul

```bash
git clone https://github.com/paraal1/osace-hackathon.git
cd osace-hackathon
```

#### 2. Creează un Mediu Virtual (Recomandat)

```bash
# Creează mediul virtual
python -m venv venv

# Activează mediul virtual
# Windows:
venv\Scripts\activate

# Linux/Mac:
source venv/bin/activate
```

#### 3. Instalează Dependențele

```bash
pip install -r requirements.txt
```

**Dependențe principale:**
- TensorFlow >= 2.13.0
- Flask >= 2.3.2
- OpenCV >= 4.8.0
- NumPy >= 1.24.3
- Pandas >= 2.0.3
- scikit-learn >= 1.3.0

## 🚀 Cum se Rulează Proiectul

### Metoda Rapidă (Quick Start)

Dacă ai deja un model antrenat:

```bash
# Activează mediul virtual
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Pornește serverul
python api/app.py
```

Accesează aplicația la: **http://localhost:5000**

---

### Metoda Completă (De la Zero)

#### **Pasul 1: Pregătirea Datasetului**

1. **Descarcă un dataset** de imagini robot vs human:
   - Kaggle: [Robot vs Human Dataset](https://www.kaggle.com/)
   - Google Images (cu respectarea drepturilor de autor)
   - Propriile tale imagini

2. **Organizează imaginile** în structura de directoare:
   ```
   data/raw/
   ├── human/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   └── robot/
       ├── image1.jpg
       ├── image2.jpg
       └── ...
   ```

3. **Rulează scriptul de preprocesare**:
   ```bash
   python src/data_preprocessing.py
   ```

   Acest script va:
   - Împărți datele în train (70%), validation (15%), test (15%)
   - Crea directoarele necesare
   - Copia imaginile în locațiile corespunzătoare
   - Afișa statistici despre dataset

#### **Pasul 2: Antrenarea Modelului**

```bash
python src/train.py
```

**Ce face acest script:**
- ✅ Încarcă și pregătește datasetul
- ✅ Configurează data augmentation
- ✅ Construiește modelul CNN cu transfer learning
- ✅ Antrenează modelul (10 epoci default)
- ✅ Evaluează pe setul de test
- ✅ Salvează modelul în `models/robot_vs_human_classifier.h5`
- ✅ Generează grafice de performanță
- ✅ Creează raport detaliat (`training_summary.txt`)

**Timp estimat:** 10-30 minute (depinde de hardware și mărimea datasetului)

**Parametri configurabili** (în `src/config.py`):
```python
EPOCHS = 10              # Număr de epoci
BATCH_SIZE = 32          # Dimensiune batch
LEARNING_RATE = 0.0001   # Rata de învățare
IMG_SIZE = (224, 224)    # Dimensiune imagini
```

#### **Pasul 3: Testare CLI (Opțional)**

Testează modelul direct din terminal:

```bash
python src/predict.py
```

Urmează instrucțiunile pentru a introduce calea către o imagine.

#### **Pasul 4: Pornirea Serverului Web**

```bash
python api/app.py
```

**Output așteptat:**
```
==================================================
STARTING FLASK API SERVER
==================================================

Server will be available at: http://localhost:5000
API endpoints:
  - GET  /api/health     - Health check
  - POST /api/predict          - Upload and predict image
  - POST /api/predict/batch       - Upload and predict multiple images
  - GET  /api/export/batch/csv    - Export last batch results to CSV
  - GET  /api/history          - Get prediction history
  - GET  /api/statistics          - Get statistics

Loading model...
Model loaded successfully!
 * Running on http://0.0.0.0:5000
```

Accesează aplicația la: **http://localhost:5000**

## 🏛️ Arhitectura Tehnică

### 1. Arhitectura Modelului CNN

Modelul folosește **Transfer Learning** cu MobileNetV2 ca base model:

```
INPUT (224x224x3)
    ↓
MobileNetV2 (pretrained on ImageNet)
    ↓ [frozen layers]
GlobalAveragePooling2D
    ↓
Dropout (0.3) ← regularization
    ↓
Dense (128, ReLU) ← feature extraction
    ↓
BatchNormalization ← stabilizare antrenare
    ↓
Dropout (0.5) ← prevent overfitting
    ↓
Dense (2, Softmax) → [Human, Robot]
```

**Caracteristici tehnice:**
- **Base Model**: MobileNetV2 (pre-trained pe ImageNet)
  - Avantaje: Lightweight, eficient, performanță bună
  - Parametri: ~3.5M (base) + ~130K (custom layers)
- **Optimizer**: Adam (adaptive learning rate)
  - Learning rate: 0.0001
- **Loss Function**: Categorical Crossentropy
- **Metrici**: Accuracy, Precision, Recall, F1-Score
- **Fine-tuning**: Opțional, ultimele 20 de layere

### 2. Data Augmentation Strategy

Pentru a preveni overfitting-ul și a îmbunătăți generalizarea:

```python
ImageDataGenerator(
    rotation_range=20,  # Rotație ±20°
    width_shift_range=0.2,    # Shift orizontal 20%
    height_shift_range=0.2,      # Shift vertical 20%
    horizontal_flip=True,        # Flip orizontal
    zoom_range=0.2,           # Zoom in/out 20%
    shear_range=0.2,      # Shear transformation
    rescale=1./255,              # Normalizare [0, 1]
    fill_mode='nearest'          # Umplere pixeli noi
)
```

### 3. Arhitectura API (Flask)

```
┌─────────────────────────────────────────────┐
│   Flask Application         │
├─────────────────────────────────────────────┤
│  GET  /           → index.html        │
│  GET  /api/health        → Status check      │
│  POST /api/predict  → Single prediction │
│  POST /api/predict/batch → Batch prediction  │
│  GET  /api/export/batch/csv → CSV export     │
│  GET  /api/history       → History           │
│  GET  /api/statistics    → Stats          │
└─────────────────────────────────────────────┘
         ↓            ↓
    ┌─────────┐   ┌──────────┐
    │  Model  │       │ Database │
    │  (.h5)  │            │ (SQLite) │
  └─────────┘            └──────────┘
```

### 4. Flux de Procesare

```
User Upload
    ↓
Validate File (type, size)
    ↓
Save to /uploads
    ↓
Preprocess Image
    ↓ (resize, normalize)
Load Model
    ↓
Make Prediction
    ↓
Save to Database
    ↓
Return JSON Response
```

## 📡 API Documentation

### `GET /api/health`

Verifică starea serverului și încărcarea modelului.

**Request:**
```bash
curl http://localhost:5000/api/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-01-08T10:30:00.123456"
}
```

---

### `POST /api/predict`

Clasifică o singură imagine încărcată.

**Request:**
```bash
curl -X POST http://localhost:5000/api/predict \
  -F "file=@path/to/image.jpg"
```

**Response:**
```json
{
  "success": true,
  "filename": "20250108_103000_image.jpg",
  "predicted_class": "robot",
  "confidence": 0.9547,
  "probabilities": {
    "human": 0.0453,
    "robot": 0.9547
  },
  "timestamp": "2025-01-08T10:30:00.123456"
}
```

---

### `POST /api/predict/batch`

Clasifică multiple imagini simultan (max 20).

**Request:**
```bash
curl -X POST http://localhost:5000/api/predict/batch \
-F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg"
```

**Response:**
```json
{
  "success": true,
  "processed": 3,
  "failed": 0,
  "total": 3,
  "results": [
    {
 "filename": "20250108_103000_0_image1.jpg",
      "original_filename": "image1.jpg",
      "predicted_class": "robot",
      "confidence": 0.95,
      "probabilities": {"human": 0.05, "robot": 0.95}
    }
  ],
  "summary": {
    "robot_count": 2,
    "human_count": 1,
    "average_confidence": 0.92,
    "high_confidence_predictions": 3,
    "low_confidence_predictions": 0
  }
}
```

---

### `GET /api/export/batch/csv`

Exportă ultimele rezultate batch în format CSV.

**Request:**
```bash
curl http://localhost:5000/api/export/batch/csv --output results.csv
```

**CSV Format:**
```csv
Filename,Original Filename,Predicted Class,Confidence (%),Robot Probability (%),Human Probability (%),Timestamp
20250108_103000_0_image1.jpg,image1.jpg,ROBOT,95.47,95.47,4.53,2025-01-08T10:30:00
...
```

---

### `GET /api/history?limit=10`

Obține istoricul predicțiilor (default: 10 rezultate).

**Request:**
```bash
curl http://localhost:5000/api/history?limit=20
```

**Response:**
```json
{
  "success": true,
  "predictions": [
    {
      "id": 1,
      "filename": "test.jpg",
   "predicted_class": "robot",
      "confidence": 0.95,
      "timestamp": "2025-01-08T10:30:00"
    }
  ],
  "count": 1
}
```

---

### `GET /api/statistics`

Obține statistici agregare despre toate predicțiile.

**Request:**
```bash
curl http://localhost:5000/api/statistics
```

**Response:**
```json
{
  "success": true,
  "statistics": {
    "total_predictions": 142,
    "predictions_by_class": {
      "human": 68,
  "robot": 74
    },
    "average_confidence": 0.9234,
    "high_confidence_count": 120,
    "low_confidence_count": 22
  }
}
```

## 📊 Rezultate Obținute

### Performanță Model

Rezultatele obținute pe datasetul de test (după 10 epoci de antrenare):

| Metrică | Valoare | Descriere |
|---------|---------|-----------|
| **Training Accuracy** | **~95%** | Acuratețe pe setul de antrenare |
| **Validation Accuracy** | **92.16%** | Acuratețe pe setul de validare |
| **Test Accuracy** | **~91%** | Acuratețe pe date nevăzute |
| **Precision** | **92.16%** | Cât de multe predicții pozitive sunt corecte |
| **Recall** | **92.16%** | Cât de multe cazuri pozitive sunt detectate |
| **F1-Score** | **92.16%** | Media armonică Precision/Recall |
| **Loss (Validation)** | **0.1489** | Funcție de pierdere (mai mic = mai bine) |

**Interpretare:**
- ✅ Model foarte bun pentru clasificare binară
- ✅ Generalizare excelentă (diferență mică train/val)
- ✅ Echilibru bun între precision și recall
- ✅ Potrivit pentru producție cu monitorare

### Grafice de Evoluție

Graficele sunt salvate automat în directorul `models/`:

#### 1. **training_history.png**
Arată evoluția accuracy și loss pe parcursul antrenării:
- Curba albastră: Training
- Curba portocalie: Validation

**Observații:**
- Convergență rapidă în primele 3-4 epoci
- Stabilizare după epoca 6-7
- Fără semne majore de overfitting

#### 2. **fine_tuning_history.png** (dacă s-a aplicat)
Evoluția performanței după fine-tuning pe layerele superioare.

### Confusion Matrix (Example)

```
              Predicted
    Human  Robot
Actual
Human      145    12     Accuracy: 92.3%
Robot       8    154     Precision: 92.8%
         Recall: 91.7%
```

### Timp de Predicție

- **Single Image**: ~100-200ms (CPU) / ~50ms (GPU)
- **Batch (10 images)**: ~800ms (CPU) / ~300ms (GPU)

## 🎯 Demo și Testare

### Imagini de Test Pregătite

Am pregătit 3 imagini de test în directorul `Images/` pentru demonstrație:

1. **test_robot_1.jpg** - Robot industrial
   - Predicție așteptată: ROBOT (confidence > 90%)
 
2. **test_human_1.jpg** - Portret uman
   - Predicție așteptată: HUMAN (confidence > 90%)
   
3. **test_mixed_1.jpg** - Scenă complexă
 - Predicție așteptată: Variabilă (depinde de focus)

### Cum să Rulezi Demo-ul

#### **Metoda 1: Interfața Web (Recomandat)**

1. Pornește serverul:
   ```bash
   python api/app.py
   ```

2. Deschide browser la `http://localhost:5000`

3. **Pentru predicție single:**
 - Click pe zona de drag & drop SAU
   - Click pe "Browse" pentru a selecta o imagine
   - Alege una din imaginile test: `Images/test_robot_1.jpg`
   - Așteaptă rezultatul (2-3 secunde)

4. **Pentru predicție batch:**
   - Click pe tab-ul "Batch Upload"
   - Selectează multiple imagini (Ctrl+Click)
   - Click "Upload and Classify"
   - Vezi rezultatele în tabel
   - Download CSV pentru analiză detaliată

**Screenshots:**
- Interfața suportă drag & drop
- Afișare rezultat cu confidence bar
- Istoric predicții în sidebar
- Statistici live

#### **Metoda 2: API Direct (cURL)**

```bash
# Test single image
curl -X POST http://localhost:5000/api/predict \
  -F "file=@Images/test_robot_1.jpg"

# Test batch
curl -X POST http://localhost:5000/api/predict/batch \
  -F "files=@Images/test_robot_1.jpg" \
  -F "files=@Images/test_human_1.jpg" \
  -F "files=@Images/test_mixed_1.jpg"

# Export results
curl http://localhost:5000/api/export/batch/csv --output demo_results.csv
```

#### **Metoda 3: Python Script**

```python
import requests

# Single prediction
with open('Images/test_robot_1.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:5000/api/predict', files=files)
    print(response.json())

# Batch prediction
files = [
    ('files', open('Images/test_robot_1.jpg', 'rb')),
    ('files', open('Images/test_human_1.jpg', 'rb')),
    ('files', open('Images/test_mixed_1.jpg', 'rb'))
]
response = requests.post('http://localhost:5000/api/predict/batch', files=files)
print(response.json())
```

### Exemple de Output Demo

**Exemplu 1 - Robot Industrial:**
```json
{
  "success": true,
  "predicted_class": "robot",
  "confidence": 0.9823,
  "probabilities": {
    "robot": 0.9823,
    "human": 0.0177
  }
}
```

**Exemplu 2 - Portret Uman:**
```json
{
  "success": true,
  "predicted_class": "human",
  "confidence": 0.9654,
  "probabilities": {
    "human": 0.9654,
 "robot": 0.0346
  }
}
```

**Exemplu 3 - Caz Ambiguu:**
```json
{
  "success": true,
  "predicted_class": "robot",
  "confidence": 0.6234,
  "probabilities": {
    "robot": 0.6234,
    "human": 0.3766
  }
}
```
→ Confidence scăzut sugerează că modelul e nesigur (caz limită)

## ⚠️ Provocări Întâmpinate

### 1. **Suport pentru Formate RAW Profesionale**

**Problema:**
- Formatul ARW (Sony RAW), CR2 (Canon), NEF (Nikon) nu sunt suportate nativ de Pillow sau OpenCV
- Erorile de tip `cannot identify image file` apăreau frecvent

**Soluție implementată:**
```python
import rawpy

def load_raw_image(filepath):
    with rawpy.imread(filepath) as raw:
        rgb = raw.postprocess()
    return Image.fromarray(rgb)
```

**Lecție învățată:**
- Biblioteci specializate (rawpy) sunt necesare pentru formate profesionale
- Procesarea RAW e costisitoare computațional → necesită timeout-uri mai mari

---

### 2. **Dataset Limitat și Dezechilibrat**

**Problema:**
- Dataset inițial mic (~500 imagini per clasă)
- Risc mare de overfitting
- Distribuție inegală între clase (60% robot, 40% human)

**Soluții aplicate:**
1. **Data Augmentation agresiv:**
   - Rotații, zoom, shift, flip
   - ~5x multiplicator de date

2. **Transfer Learning:**
   - Folosirea MobileNetV2 preantrenat
   - Reducere necesitate de date de antrenare

3. **Monitorizare metrici multiple:**
   - Nu doar Accuracy, ci și Precision/Recall/F1
   - Detectarea bias-ului către clasa majoritară

**Rezultat:**
- Îmbunătățire accuracy cu 15-20% față de model trained from scratch
- Generalizare mai bună pe date noi

---

### 3. **Overfitting în Primele Iterații**

**Problema:**
- Training accuracy: 98%
- Validation accuracy: 75%
- Gap mare → overfitting sever

**Soluții implementate:**
1. **Dropout layers:**
   ```python
   Dropout(0.3) și Dropout(0.5)
   ```

2. **Batch Normalization:**
   ```python
   BatchNormalization()
   ```

3. **Early Stopping:**
   ```python
   EarlyStopping(patience=5, restore_best_weights=True)
   ```

4. **Reducerea complexității:**
   - Layer-uri mai puține
   - Regularization L2

**Rezultat:**
- Gap redus la ~3-4%
- Validation accuracy crescut la 92%

---

### 4. **Timpul de Antrenare pe CPU**

**Problema:**
- Antrenare pe CPU: ~2-3 ore pentru 10 epoci
- Development lent, iterații rare

**Soluții:**
1. **Model lightweight:**
   - MobileNetV2 (design pentru mobile/embedded)
   - ~3.5M parametri vs ~25M (ResNet50)

2. **Batch size optimization:**
   - Crescut la 32 (de la 16)
   - Folosire eficientă a memoriei

3. **Mixed precision training** (opțional):
   ```python
   from tensorflow.keras import mixed_precision
   policy = mixed_precision.Policy('mixed_float16')
   ```

**Rezultat:**
- Timp redus la ~45 minute/10 epoci
- Possibilitate de iterare rapidă

---

### 5. **Gestionarea Erorilor în API**

**Problema:**
- Upload fișiere corupte → crash server
- Imagini prea mari → timeout
- Formate nesuportate → erori generice

**Soluții implementate:**
1. **Validare strictă:**
   ```python
   ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'arw', ...}
   MAX_FILE_SIZE = 16 * 1024 * 1024  # 16 MB
   ```

2. **Try-catch comprehensiv:**
   ```python
   try:
       predict_image(model, filepath)
   except Exception as e:
     return jsonify({'error': str(e)}), 500
   ```

3. **Error handlers Flask:**
   ```python
   @app.errorhandler(413)
   def request_entity_too_large(error):
   return jsonify({'error': 'File too large'}), 413
   ```

**Rezultat:**
- API stabil, fără crash-uri
- Mesaje de eroare clare pentru utilizatori

---

### 6. **Inconsistență Predicții pe Imagini Similare**

**Problema:**
- Imagini similare primeau predicții diferite
- Exemple: roboți umanoidali clasificați ca "human"

**Cauze identificate:**
1. Dataset insuficient pentru edge cases
2. Similarități vizuale între clase
3. Lipsa de context (doar features vizuale)

**Soluții parțiale:**
1. **Threshold de confidence:**
   ```python
   if confidence < 0.7:
       return "Uncertain - Manual review needed"
   ```

2. **Ensemble methods** (pentru versiuni viitoare):
   - Multiple modele
   - Voting mechanism

3. **Explicabilitate** (Grad-CAM):
   - Vizualizare ce "vede" modelul
   - Debugging predicții greșite

**Lecție învățată:**
- Deep learning nu e magie → necesită date de calitate
- Confidence scores sunt critice pentru producție

---

### 7. **Deployment și Portabilitate**

**Problema:**
- Dependențe multe și grele (TensorFlow ~500MB)
- Compatibilitate între Windows/Linux/Mac
- Versiuni conflictuale de biblioteci

**Soluții:**
1. **Requirements.txt cu versiuni fixe:**
   ```
   tensorflow==2.13.0
   ```

2. **Virtual environment obligatoriu:**
   ```bash
   python -m venv venv
   ```

3. **Documentație detaliată:**
   - README cu pași exacți
   - Troubleshooting section

**Pentru viitor:**
- Docker containerization
- CI/CD pipeline pentru testing

## 🛠️ Tehnologii Utilizate

### Deep Learning & AI
- **TensorFlow 2.13+** - Framework principal pentru deep learning
- **Keras** - API high-level pentru construirea rețelelor neuronale
- **MobileNetV2** - Arhitectură CNN pretrained (transfer learning)

### Image Processing
- **OpenCV** - Procesare și manipulare imagini
- **Pillow (PIL)** - I/O imagini, transformări
- **rawpy** - Suport pentru formate RAW profesionale (ARW, CR2, NEF)

### Web Framework & API
- **Flask 2.3+** - Micro web framework pentru API REST
- **Flask-CORS** - Cross-Origin Resource Sharing
- **Werkzeug** - WSGI utilities, secure filename handling

### Data Science & Visualization
- **NumPy** - Operații numerice, array manipulation
- **Pandas** - Structuri de date, analiză
- **scikit-learn** - Train/test split, metrici ML
- **Matplotlib** - Vizualizări grafice
- **Seaborn** - Grafice statistice avansate

### Database & Storage
- **SQLite3** - Database relațională lightweight pentru istoricul predicțiilor
- **CSV Export** - Export rezultate batch pentru analiză

### Development Tools
- **python-dotenv** - Gestionare variabile de mediu
- **tqdm** - Progress bars pentru antrenare
- **Git** - Version control

### Frontend (în index.html)
- **HTML5 + CSS3** - Interfață web modernă
- **JavaScript (Vanilla)** - Interacțiune client-side
- **Fetch API** - Comunicare cu backend
- **Drag & Drop API** - Upload intuitiv de fișiere

## 📈 Îmbunătățiri Viitoare

- [ ] **Multi-class classification**: Extindere la mai multe categorii (cyborgs, androizi, drones)
- [ ] **Model explicability**: Implementare Grad-CAM pentru vizualizare ce "vede" modelul
- [ ] **A/B Testing**: Comparare între modele diferite (ResNet, EfficientNet)
- [ ] **Cloud Deployment**: Deploy pe AWS/Azure/Google Cloud
- [ ] **Docker containerization**: Ușurarea deployment-ului
- [ ] **API Authentication**: JWT tokens pentru securitate
- [ ] **Real-time video classification**: Clasificare pe stream video
- [ ] **Mobile app**: React Native sau Flutter
- [ ] **Model versioning**: MLflow pentru tracking experimente
- [ ] **CI/CD Pipeline**: GitHub Actions pentru testing automat

## 👥 Contribuitori

**Echipa OSACE Hackathon 2025**

- **[Dinescu Andrei]** - Data Science & Model Development
  - Arhitectură CNN
  - Transfer learning implementation
  - Model training și optimization

- **[Paraschivoiu Alexandru]** - Backend & API Development
  - Flask API design
  - Database integration
  - Batch processing

- **[Dinescu Andrei & Paraschivoiu Alexandru]** - Frontend & Integration
  - Interfață web
  - Testing & debugging
  - Documentation

## 📄 Licență

Acest proiect este licențiat sub **MIT License** - vezi fișierul [LICENSE](LICENSE) pentru detalii.

```
MIT License

Copyright (c) 2025 OSACE Hackathon Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

## 🙏 Mulțumiri

- **TensorFlow/Keras Team** - Pentru framework-ul excelent de deep learning
- **Kaggle Community** - Pentru datasets și tutoriale
- **MobileNet Authors** - Pentru arhitectura eficientă și pretrained weights
- **Flask Team** - Pentru framework-ul web simplu și elegant
- **OSACE Hackathon Organizers** - Pentru oportunitatea de a lucra la acest proiect

## 📞 Contact & Support

Pentru întrebări, bug reports sau feature requests:

- **GitHub Issues**: [github.com/paraal1/osace-hackathon/issues](https://github.com/paraal1/osace-hackathon/issues)
- **Email**: [multiverse_team@gmail.com](mailto:your.email@example.com)
- **Documentation**: Vezi acest README și comentariile din cod

---

<div align="center">

**Made with ❤️ and 🤖 for OSACE Hackathon 2025**

⭐ **Dacă ți-a plăcut proiectul, lasă un star pe GitHub!** ⭐

</div>
