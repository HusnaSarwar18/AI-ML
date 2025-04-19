# 🧠 AI & Deep Learning Projects Portfolio by Husna Sarwar

This repository contains three end-to-end intelligent systems that demonstrate my practical understanding of Artificial Intelligence, Deep Learning, and Data Processing. Each project showcases unique capabilities with complete pipelines for data processing, model training, and inference.

---

## 📘 1. RAG (Retrieval-Augmented Generation) Application with Gemini Input Selection

An interactive RAG-based assistant designed specifically for users **under 18**, powered by Google's **Gemini-1.5-Flash** model with advanced embedding and retrieval techniques.

### 🔁 Workflow

Upload File → Extract Text → Vectorize → Retrieve → Generate → Speak/Display → Cleanup

### 🎯 Features

- **Input Options**:
  - Upload **PDF**: Extracts text using `PyPDFLoader`
  - Upload **Audio**: Converts to text using `SpeechRecognition`

- **Processing**:
  - Text is chunked using `RecursiveCharacterTextSplitter`
  - Embedded using Gemini's `embedding-001` model
  - Stored in `ChromaDB` (vector database)

- **Query Handling**:
  - User submits a query
  - Relevant chunks retrieved based on embedding similarity
  - Gemini-1.5-Flash generates a concise response (max **3 sentences**)

- **Output**:
  - Text response
  - Audio response generated using `gTTS` with **language auto-detection**
  - Optional cleanup of uploaded files

- **Security Feature**:
  - Only responds to users **below 18 years of age**

---

## 🔢 2. Handwritten Number Classification with PyTorch

A deep learning pipeline to classify handwritten digits with support for handling **class imbalance**, model performance tracking, and architecture comparison.

### 🧩 Workflow

Load Data → Calculate Weights → Train Model → Evaluate → Save

### 🔨 Features

- **Data Preparation**:
  - Loads data from `train/` and `test/` directories
  - Applies resizing (128x128), normalization, and tensor conversion
  - Calculates inverse-frequency weights to handle class imbalance

- **Model Architectures**:
  - **Simple CNN**: 3 convolutional layers + 2 fully connected layers
  - **ResNet-18**: Pre-trained with a modified output layer

- **Training**:
  - Loss Function: **Weighted CrossEntropyLoss**
  - Optimizer: **Adam** with learning rate `0.001`
  - Tracks accuracy and loss per epoch
  - Includes early stopping mechanism

- **Evaluation**:
  - Metrics: Accuracy, Precision, Recall, F1-Score
  - Confusion matrix visualization
  - Trained model saved as `model.pth`

- **Comparison Table**:

| Model      | Accuracy | Training Time |
|------------|----------|----------------|
| Simple CNN | ~92%     | Faster         |
| ResNet-18  | ~96%     | Slower         |

- **Extras**:
  - GPU acceleration support
  - Modular and reusable code design

---

## 🧾 3. Score Extraction from Exam Sheets

An end-to-end pipeline for recognizing, classifying, and calculating handwritten and printed scores from table images.

### 📊 Workflow

Drive Mount → Data Extract → Model Train → Table Segment → OCR & Digit Recognition → Score Calculation → Output Generation

### ⚙️ Features

- **Setup**:
  - Mounts Google Drive (for dataset access)
  - Extracts and prepares dataset
  - Resizes images to 64x64, normalizes, and converts to tensors

- **Modeling**:
  - Two architectures: Simple CNN and ResNet-18
  - Handles class imbalance with inverse-frequency weights
  - Trained using **Weighted CrossEntropy + Adam Optimizer**

- **Table Processing**:
  - **OCR Module**: Uses `Tesseract` to extract printed text
  - **Segmentation**: Edge detection and contour analysis to locate table cells
  - **Prediction**: Classifies digits in each cell using CNN
  - **Calculation**: Totals computed for each table

- **Output**:
  - Annotated images with predicted digits and total scores
  - Text files summarizing OCR results  
  - Sample Output:
    ```
    Image 1: table1.png → Total Score: 87  
    Image 2: table2.png → Total Score: 92
    ```

- **Performance**:
  - Processes 100+ images in under 5 minutes (using Colab GPU)
  - Digit classification accuracy: **92-96%**
  - OCR accuracy: ~85% (depending on image quality)

---

## 🛠️ Requirements

- **Python** 3.10+
- **Google Colab** (for GPU acceleration)
- Required Libraries:
  - `PyTorch`, `torchvision`
  - `OpenCV`, `PIL`, `matplotlib`
  - `Tesseract OCR`
  - `SpeechRecognition`, `gTTS`
  - `Langchain`, `ChromaDB`
  - Gemini API access

---

## 📁 Repository Structure

```bash
.
├── RAG_Gemini_Assistant/
│   ├── main.py
│   ├── utils.py
│   ├── requirements.txt
│   └── README.md
├── Digit_Classifier/
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── data/
│   └── README.md
├── ScoreTable_Processor/
│   ├── segment.py
│   ├── ocr_predict.py
│   ├── total_calculator.py
│   └── README.md
└── README.md  ← This File



👩‍💻 Author
Husna Sarwar
🎓 Computer Science Student @ PUCIT
💡 Passionate about Deep Learning, Python, and Building Real-World AI Solutions