# 🍇 DateFruit Classification using ANN

This project implements an **Artificial Neural Network (ANN)** in PyTorch to classify different varieties of Date Fruits based on their physical and chemical attributes.

## 📂 Dataset
- **Source**: `DateFruit_Dataset.csv`
- **Target Column**: `Class` (7 unique fruit classes)
- **Features**: Numerical attributes like size, weight, moisture, etc.

## ⚙️ Workflow
1. Data Preprocessing
   - Label Encoding for target classes
   - Standardization using `StandardScaler`
   - Train/Test split (70/30)

2. Model Architecture
   - Input Layer: `X.shape[1]` features
   - Hidden Layers: Two layers with 64 neurons each, ReLU activation
   - Output Layer: 7 neurons (multi-class classification)

3. Training
   - Loss: `CrossEntropyLoss`
   - Optimizer: `Adam`
   - Epochs: 100
   - Batch Size: 32

4. Evaluation
   - Accuracy calculated on test set
   - Prints total samples, correct predictions, and accuracy %

## 📊 Results
- Achieved 93.4% accuracy on test data 
- Model generalizes well across multiple fruit classes.

## 🚀 How to Run
```bash
git clone https://github.com/<your-username>/DateFruit-ANN.git
cd DateFruit-ANN
pip install -r requirements.txt
python ann_datefruit.py
