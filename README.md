# Diabetic Retinopathy Classification

## Overview
This repository provides an **end-to-end deep learning pipeline** for detecting **Diabetic Retinopathy (DR)** from **fundus images**. The model achieves **80% accuracy** using **DenseNet-121 and a custom CNN**. The project includes **data preprocessing, model training, evaluation, and testing**, with automated classification and performance analysis.

---

## Project Structure

```
Diabetic-Retinopathy-Classification/
│── classfi.py                  # Classifies images into severity folders
│── check.py                    # Verifies classification correctness
│── training_file.py             # Trains the DenseNet-121 & CNN model
│── testing111.py                # Performs inference on test images
│── classification_report.csv    # Evaluation report
│── predictions.csv              # Model predictions & confidence scores
│── dataset/                     # Folder for original dataset (if applicable)
│── README.md                    # Project Documentation
|── datset and few points
```

## Dataset
The dataset consists of **retinal fundus images**, each labeled by clinicians for **Diabetic Retinopathy (DR) severity**. The dataset used in this project comes from the **APTOS 2019 Blindness Detection dataset** and is structured as follows:

### **Classes in the Dataset:**
- **0 - No DR** (Healthy retina)
- **1 - Mild DR** (Few microaneurysms present)
- **2 - Moderate DR** (More blood vessel abnormalities)
- **3 - Severe DR** (Many hemorrhages and venous beading)
- **4 - Proliferative DR** (Advanced DR, new blood vessel formation)

Each image is categorized into respective class-based folders and used for **training, validation, and testing**.

---

## Setup & Installation

### **1. Clone Repository**
```bash
git clone https://github.com/RAJEEVRANJAN0001/Diabetic-Retinopathy-Classification.git
cd Diabetic-Retinopathy-Classification
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Prepare Dataset**
Ensure your dataset is structured as follows:
```
classified_images/
│── 0_No_DR/
│── 1_Mild/
│── 2_Moderate/
│── 3_Severe/
│── 4_Proliferative_DR/
```
Run `classfi.py` to organize images:
```bash
python classfi.py
```

Run `check.py` to verify classification:
```bash
python check.py
```

---

## Code Explanation

### **1. Image Classification (`classfi.py`)**
- Reads the dataset labels (`train.csv`)
- Organizes fundus images into severity-based folders
- Moves images into respective classes for easy training

### **2. Dataset Verification (`check.py`)**
- Cross-checks the classification to ensure images are correctly labeled
- Generates a `classification_report.csv` file

### **3. Model Training (`training_file.py`)**
This script implements **DenseNet-121** with a **custom CNN** to classify DR severity levels.
- Uses **ImageDataGenerator** for data augmentation
- Transfers learning from **DenseNet-121** (pretrained on ImageNet)
- Adds CNN layers for better feature extraction
- Optimizes using **Adam optimizer** with **Cohen’s Kappa Score** for evaluation
- Saves the trained model as `diabetic_retinopathy_model.h5`

Run training with:
```bash
python training_file.py
```

### **4. Model Testing (`testing111.py`)**
- Loads `diabetic_retinopathy_model.h5`
- Predicts severity for test images
- Saves predictions in `predictions.csv`

Run predictions with:
```bash
python testing111.py
```

---

## Model Training Details
- **Model Used:** DenseNet-121 with additional CNN layers
- **Epochs:** 50
- **Batch Size:** 32
- **Optimizer:** Adam (learning rate = 0.0001)
- **Loss Function:** Sparse Categorical Crossentropy
- **Performance Metric:** Cohen’s Kappa Score

The model is trained using **transfer learning**, allowing faster convergence and better accuracy.

---

## Model Testing & Predictions
Run `testing111.py` to classify new retina images:
```bash
python testing111.py
```
This generates `predictions.csv` with:
- **Filename**
- **Predicted Class**
- **Confidence Score**

---

## Evaluation Metrics
### **Achieved Accuracy: 80%**


### **View Classification Report**
```bash
cat classification_report.csv
```

---

## Results
- ✅ **Automated Image Classification**
- ✅ **Deep Learning Model (DenseNet-121 + CNN)**
- ✅ **80% Accuracy Achieved**
- ✅ **End-to-End DR Detection Pipeline**

---

## Future Improvements
- Fine-tune the CNN architecture for better feature extraction.
- Implement ensemble learning to improve classification.
- Integrate real-time DR detection in medical applications.

---

## Contributors
Developed by **Rajeev Ranjan Pratap Singh** 🚀

Feel free to contribute and improve the model!

---

## License
This project is licensed under the **MIT License**.
