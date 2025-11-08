# ?? Robot vs Human Image Classifier

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Un clasificator de imagini bazat pe CNN (Convolutional Neural Network) care distinge între **robo?i** ?i **oameni**. Proiectul include preprocesarea datelor, antrenarea modelului, un API Flask ?i o interfa?? web pentru testare.

## ?? Cuprins

- [Caracteristici](#-caracteristici)
- [Arhitectura Proiectului](#-arhitectura-proiectului)
- [Instalare](#-instalare)
- [Utilizare](#-utilizare)
- [Arhitectura Modelului](#-arhitectura-modelului)
- [API Endpoints](#-api-endpoints)
- [Rezultate](#-rezultate)
- [Provoc?ri](#-provoc?ri)
- [Contribuitori](#-contribuitori)

## ? Caracteristici

- ? **Transfer Learning** cu MobileNetV2 preantrenat pe ImageNet
- ? **Data Augmentation** pentru îmbun?t??irea performan?ei
- ? **API REST** cu Flask pentru predic?ii
- ? **Interfa?? Web** intuitiv? cu drag & drop
- ? **Persisten?? Date** cu SQLite pentru istoricul predic?iilor
- ? **Suport RAW** pentru formate ARW, CR2, NEF, etc.
- ? **Metrici Complete**: Accuracy, Precision, Recall, F1-Score
- ? **Visualiz?ri** pentru evolu?ia antren?rii

## ??? Arhitectura Proiectului

```
osace-hackathon/
??? data/
?   ??? raw/          # Dataset original
?   ?   ??? human/        # Imagini cu oameni
?   ?   ??? robot/        # Imagini cu robo?i
?   ??? train/            # Date de antrenare (70%)
?   ??? val/    # Date de validare (15%)
?   ??? test/ # Date de testare (15%)
??? models/
?   ??? robot_vs_human_classifier.h5  # Model salvat
?   ??? training_history.png        # Grafic antrenare
?   ??? training_summary.txt          # Sumar rezultate
??? src/
?   ??? config.py   # Configur?ri
?   ??? data_preprocessing.py  # Preprocesare date
?   ??? model.py      # Arhitectur? CNN
?   ??? train.py               # Script antrenare
?   ??? predict.py       # Script predic?ii
??? api/
?   ??? app.py# Flask API
?   ??? database.py            # SQLite database
?   ??? templates/
?       ??? index.html     # Interfa?? web
??? uploads/               # Imagini înc?rcate
??? Images/           # Imagini de test
??? requirements.txt           # Dependen?e Python
??? predictions.db      # Baza de date SQLite
??? README.md                  # Documenta?ie
```

## ?? Instalare

### Prerequisite

- Python 3.8 sau mai nou
- pip (Python package manager)
- Git

### Pa?i de instalare

1. **Cloneaz? repository-ul**

```bash
git clone https://github.com/paraal1/osace-hackathon.git
cd osace-hackathon
```

2. **Creeaz? un mediu virtual (op?ional, dar recomandat)**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Instaleaz? dependen?ele**

```bash
pip install -r requirements.txt
```

## ?? Utilizare

### Pasul 1: Preg?tirea Datasetului

1. **Descarc? un dataset** de imagini robot vs human (ex: de pe Kaggle)
2. **Organizeaz? imaginile** în structura:
   ```
   data/raw/
   ??? human/
   ?   ??? image1.jpg
   ?   ??? image2.jpg
   ?   ??? ...
   ??? robot/
       ??? image1.jpg
       ??? image2.jpg
       ??? ...
   ```

3. **Ruleaz? preprocesarea**:
   ```bash
   python src/data_preprocessing.py
   ```

### Pasul 2: Antrenarea Modelului

```bash
python src/train.py
```

Acest script va:
- Împ?r?i datele în train/val/test
- Antrena modelul CNN
- Salva modelul în `models/robot_vs_human_classifier.h5`
- Genera grafice de evolu?ie
- Crea un raport de performan??

**Parametri configurabili** (în `src/config.py`):
- `EPOCHS`: Num?r de epoci (default: 10)
- `BATCH_SIZE`: Dimensiune batch (default: 32)
- `LEARNING_RATE`: Rata de înv??are (default: 0.0001)
- `IMG_SIZE`: Dimensiune imagini (default: 224x224)

### Pasul 3: Testarea Modelului

**CLI Testing**:
```bash
python src/predict.py
```

### Pasul 4: Pornirea API-ului Web

```bash
python api/app.py
```

Acceseaz? interfa?a la: **http://localhost:5000**

## ?? Arhitectura Modelului

### Transfer Learning cu MobileNetV2

```
MobileNetV2 (pretrained on ImageNet)
    ?
GlobalAveragePooling2D
    ?
Dropout (0.3)
    ?
Dense (128, ReLU)
    ?
BatchNormalization
    ?
Dropout (0.5)
    ?
Dense (2, Softmax) ? [Human, Robot]
```

**Caracteristici**:
- **Base Model**: MobileNetV2 (frozen layers)
- **Optimizer**: Adam (lr=0.0001)
- **Loss Function**: Categorical Crossentropy
- **Metrici**: Accuracy, Precision, Recall
- **Fine-tuning**: Op?ional, ultimele 20 layere

### Data Augmentation

```python
- Rotation: ±20°
- Width/Height Shift: 20%
- Horizontal Flip: Yes
- Zoom: 20%
- Shear: 20%
- Normalization: [0, 1]
```

## ?? API Endpoints

### `GET /api/health`
Verific? starea serverului ?i modelului.

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-01-08T10:30:00"
}
```

### `POST /api/predict`
Clasific? o imagine înc?rcat?.

**Request**: `multipart/form-data` cu field `file`

**Response**:
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
  "timestamp": "2025-01-08T10:30:00"
}
```

### `GET /api/history?limit=10`
Ob?ine istoricul predic?iilor.

**Response**:
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

### `GET /api/statistics`
Ob?ine statistici despre predic?ii.

**Response**:
```json
{
  "success": true,
  "statistics": {
    "total_predictions": 42,
    "predictions_by_class": {
   "human": 20,
      "robot": 22
    },
    "average_confidence": 0.9234
  }
}
```

## ?? Rezultate

### Performan?? Model

| Metric? | Valoare |
|---------|---------|
| **Training Accuracy** | ~95% |
| **Validation Accuracy** | ~92% |
| **Test Accuracy** | ~91% |
| **Precision** | ~93% |
| **Recall** | ~90% |
| **F1-Score** | ~91% |

*Not?: Rezultatele pot varia în func?ie de dataset ?i parametrii de antrenare.*

### Grafice de Evolu?ie

Graficele sunt salvate automat în `models/`:
- `training_history.png` - Accuracy & Loss pentru train/val
- `fine_tuning_history.png` - Evolu?ia fine-tuning-ului

## ??? Tehnologii Utilizate

- **Deep Learning**: TensorFlow 2.13+, Keras
- **Image Processing**: OpenCV, Pillow, rawpy
- **Web Framework**: Flask 2.3+
- **Database**: SQLite3
- **Data Science**: NumPy, Pandas, scikit-learn
- **Visualization**: Matplotlib, Seaborn

## ?? Provoc?ri Întâmpinate

### 1. **Suport pentru Formate RAW**
- **Problem?**: Formatul ARW (Sony RAW) nu este suportat nativ
- **Solu?ie**: Implementare rawpy pentru conversie RAW ? RGB

### 2. **Dataset Limitat**
- **Problem?**: Dataset mic poate duce la overfitting
- **Solu?ie**: Data augmentation agresiv + Transfer Learning

### 3. **Balan?a Claselor**
- **Problem?**: Distribu?ie inegal? între clase
- **Solu?ie**: Monitorizare Precision/Recall, nu doar Accuracy

### 4. **Timpul de Antrenare**
- **Problem?**: Antrenarea de la zero e costisitoare
- **Solu?ie**: Transfer Learning cu MobileNetV2 (pretrained)

### 5. **Limit?ri Hardware**
- **Problem?**: GPU limitat pentru antrenare
- **Solu?ie**: Model u?or (MobileNetV2), batch size redus

## ?? To-Do / Îmbun?t??iri Viitoare

- [ ] Suport pentru mai multe clase (ex: cyborgs, androizi)
- [ ] Deploy pe cloud (Heroku, AWS, Azure)
- [ ] Explicabilitate model (Grad-CAM, LIME)
- [ ] API authentication pentru produc?ie
- [ ] Batch prediction pentru multiple imagini
- [ ] Model versioning ?i A/B testing
- [ ] Docker containerization
- [ ] CI/CD pipeline

## ?? Contribuitori

**Echipa Osace Hackathon**:
- **Membru 1** - Preprocesare & Dataset
- **Membru 2** - Model & Antrenare
- **Membru 3** - Integrare & Interfa??

## ?? Licen??

Acest proiect este licen?iat sub MIT License - vezi fi?ierul [LICENSE](LICENSE) pentru detalii.

## ?? Mul?umiri

- **TensorFlow/Keras** pentru framework-ul de deep learning
- **Kaggle** pentru resurse de dataset
- **MobileNet** pentru arhitectura preantrenat?
- **Flask** pentru framework-ul web simplu ?i elegant

---

**Made with ?? for OSACE Hackathon**

Pentru întreb?ri sau probleme, deschide un [GitHub Issue](https://github.com/paraal1/osace-hackathon/issues).
