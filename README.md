# 🌱 CropGuard AI – Crop Disease Detection & Advisory System

**Best AI Innovator Award Entry** | Seamedu Awards 2026

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11+-ee4c2c)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**98.63% Validation Accuracy** on real Maharashtra farm images (Soybean + Sugarcane)

---

## 🎯 Problem Statement
Farmers in Maharashtra (Pune, Satara, Ahmednagar, Solapur, Nashik) lose 20–30% of their soybean and sugarcane crops every year due to late detection of diseases. Manual inspection is slow, expensive, and not scalable.

**CropGuard AI** is a deep learning solution that identifies **13 different crop diseases** from a single leaf photo in seconds and provides localized farmer advisory.

---

## ✨ Key Features
- Real-time leaf disease classification (Soybean + Sugarcane)
- 98.63% validation accuracy
- Simple, farmer-friendly web interface (Streamlit)
- Personalized advisory for Maharashtra region
- Lightweight model (EfficientNet-B0) → easy to deploy on mobile/edge
- Fully reproducible training pipeline

---

## 📊 Results
- **Best Validation Accuracy**: **98.63%**
- Final Epoch: Train Acc 99.25% | Val Acc 96.72%
- Trained on **3,658 real farm images** collected from Maharashtra

![Loss & Accuracy Curves](reports/training_curves.png)
![Confusion Matrix](reports/confusion_matrix.png)

---

## 🛠️ Model Architecture
- **Base Model**: EfficientNet-B0 (pre-trained on ImageNet)
- **Transfer Learning**: First 6 MBConv blocks frozen
- **Classifier Head**: Dropout(0.2) + Linear(1280 → 13)
- **Input**: 224×224 RGB images
- **Total Trainable Parameters**: ~1.2 million

---

## 📁 Dataset
- **Soybean Leaf Disease Dataset** (Maharashtra farms) – Mendeley
- **Sugarcane Leaf Disease Dataset** (Pune district) – Mendeley
- Total: **3,658 images** across **13 classes** after cleaning & filtering

---

## 🚀 How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/AmarjeetJha17/CropDiseaseAI.git
cd CropDiseaseAI
```

### 2. Set up a Virtual Environment (Recommended)
```bash
python -m venv env
# On Windows:
env\Scripts\activate
# On Mac/Linux:
source env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run streamlit_app/app.py
```
The application will open automatically in your default internet browser (usually at `http://localhost:8501`).

---

## 📂 Project Structure
```text
CropDiseaseAI/
│
├── streamlit_app/      # Streamlit web application
│   └── app.py          # Main application file
│
├── notebooks/          # Machine Learning scripts and notebooks
│   ├── 01_preprocessing.py   
│   ├── 03_training.py
│   └── custom_dataset.py
│
├── models/             # Saved PyTorch model weights
├── reports/            # Training visualizations and evaluation metrics
├── data/               # Raw and processed data (ignored by git)
├── architecture.py     # Neural network architecture definition
├── requirements.txt    # Project Python dependencies
└── README.md           # Instructions and documentation
```

---

## 🤝 Contributing
Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License
Distributed under the **MIT License**.

---

## 👨‍💻 Author
**Amarjeet Jha**  
*Submission for Seamedu Awards 2026 - Best AI Innovator Category*
