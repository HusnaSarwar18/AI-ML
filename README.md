# 🧠 AI & Deep Learning Projects Portfolio by Husna Sarwar

This repository contains three end-to-end intelligent systems demonstrating my practical understanding of Artificial Intelligence, Deep Learning, and Data Processing. Each project showcases unique capabilities with complete pipelines for processing, training, and inference.

---

## 📘 1. RAG (Retrieval-Augmented Generation) Application with Gemini Input Selection

An interactive RAG-based assistant designed especially for users **under 18**, powered by Google's **Gemini-1.5-Flash** model and advanced embedding & retrieval techniques.

### 🔁 Workflow

Upload File → Extract Text → Vectorize → Retrieve → Generate → Speak/Display → Cleanup


### 🎯 Features

- **Input Options**:
  - Upload **PDF**: Extracts text via `PyPDFLoader`
  - Upload **Audio**: Converts to text using `SpeechRecognition`
  
- **Processing**:
  - Text is chunked using `RecursiveCharacterTextSplitter`
  - Embedded using Gemini's `embedding-001` model
  - Stored in `ChromaDB` vector database

- **Query Handling**:
  - User submits a query
  - Relevant chunks retrieved based on embedding similarity
  - Gemini-1.5-Flash generates concise, **max 3-sentence** response

- **Output**:
  - Text response
  - Audio response generated using `gTTS` with **language auto-detection**
  - Optional cleanup of uploaded files

- **Security Feature**:
  - Only responds to users **below 18 years old**

---

## 🔢 2. Handwritten Number Classification with PyTorch

A deep learning pipeline to classify handwritten digits with support for handling **class imbalance**, performance tracking, and model comparison.

### 🧩 Workflow

Load Data → Calculate Weights → Train Model → Evaluate → Save


### 🔨 Features

- **Data Preparation**:
  - Loads from structured `train/` and `test/` directories
  - Applies resizing (128x128), normalization, and tensor conversion
  - Calculates inverse-frequency weights for class imbalance

- **Model Architectures**:
  - **Simple CNN**: 3 Conv layers + 2 FC layers
  - **ResNet-18**: Pre-trained with modified final layer

- **Training**:
  - Uses **Weighted CrossEntropyLoss**
  - Optimized via **Adam (lr=0.001)**
  - Tracks epoch-wise accuracy and loss
  - Early stopping implemented

- **Evaluation**:
  - Accuracy, Precision, Recall, F1-Score
  - Confusion matrix visualization
  - Saves model as `model.pth`

- **Comparison**:

| Model      | Accuracy | Training Time |
|------------|----------|----------------|
| Simple CNN | ~92%     | Faster         |
| ResNet-18  | ~96%     | Slower         |

- **Extras**:
  - GPU support
  - Modular and flexible design

---

## 🧾 3. Score Extraction from Exam Sheets

An end-to-end pipeline for recognizing, classifying, and calculating handwritten and printed scores from table images.

### 📊 Workflow


Drive Mount → Data Extract → Model Train → Table Segment → OCR & Digit Recognition → Score Calculation → Output Generation


### ⚙️ Features

- **Setup**:
  - Mounts Google Drive
  - Extracts dataset zip
  - Prepares images (resize to 64x64, normalize, tensor)

- **Modeling**:
  - Two options: Simple CNN and ResNet-18
  - Handles class imbalance via inverse-frequency weights
  - Trains using **weighted cross-entropy + Adam**

- **Table Processing**:
  - **OCR Module**: Uses `Tesseract` to extract printed text
  - **Segmentation**: Edge detection + contour analysis for table cells
  - **Prediction**: CNN classifies digits in each cell
  - **Calculation**: Totals per table calculated

- **Output**:
  - Annotated output images with predicted digits and totals
  - Individual text files for OCR results
  - Sample:

    ```
    Image 1: table1.png → Total Score: 87  
    Image 2: table2.png → Total Score: 92
    ```

- **Performance**:
  - Processes 100+ images in < 5 mins (Colab GPU)
  - Digit classification accuracy: **92-96%**
  - OCR accuracy: ~85% (depends on image quality)

---

## 🛠️ Requirements

- Python 3.10+
- Google Colab (for GPU acceleration)
- Libraries:
  - PyTorch, torchvision
  - OpenCV, PIL, matplotlib
  - Tesseract OCR
  - SpeechRecognition, gTTS
  - Langchain, ChromaDB
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

🧠 Author
Husna Sarwar
🎓 Computer Science Student @ PUCIT
💡 Passionate about Deep Learning, Python, and Building Real-World AI Solutions
