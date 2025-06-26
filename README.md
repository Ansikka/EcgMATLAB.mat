# 🩺 LHOM-Based Feature Extraction on ECG Signals

A research-driven project focused on applying **Logarithmic Higher Order Moments (LHOM)** to extract meaningful features from **ECG (Electrocardiogram) signals**, 
improving the accuracy and interpretability of biomedical signal classification models.
 This project is a part of my research internship at **IIT Patna**, under the Biomedical Signal Processing domain.
> 
## 📌 Table of Contents

- [Introduction](#introduction)
- [Motivation](#motivation)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Comparative Analysis](#comparative-analysis)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)


## 🧠 Introduction

Electrocardiogram (ECG) signals are one of the most critical sources of information in diagnosing cardiac diseases. 
This project introduces the use of **Logarithmic Higher Order Moments (LHOM)** for extracting subtle features from ECG signals that are often missed by traditional linear methods.


## 🎯 Motivation

- Traditional feature extraction (FFT, Wavelets) lacks sensitivity to **non-linear** and **non-Gaussian** components.
- LHOM offers a new perspective for analyzing the **skewness, kurtosis, and non-stationarity** of ECG data.
- The goal is to improve classification performance in detecting anomalies such as **arrhythmias**.

## ⚙️ Methodology

1. **Data Acquisition**  
   - Used datasets from **MIT-BIH Arrhythmia Database** via PhysioNet.

2. **Preprocessing**  
   - Bandpass filtering (0.5–40 Hz)
   - Baseline drift removal
   - Noise filtering using median filters

3. **LHOM Feature Extraction**  
   - Implemented algorithms to compute:
     - Higher-order central moments
     - Logarithmic scale-based transformations
     - Entropy, skewness, and kurtosis

4. **Feature Selection & Classification**  
   - Dimensionality reduction (PCA)
   - Model training using SVM, Random Forest, and CNN

5. **Evaluation**  
   - Accuracy, Precision, Recall, F1-score
   - ROC-AUC and confusion matrix analysis


## 💻 Installation

Clone the repository:

git clone https://github.com/your-username/lhom-ecg-feature-extraction.git
cd lhom-ecg-feature-extraction

Install dependencies

Edit
pip install -r requirements.txt
🚀 Usage
To run the feature extraction pipeline:

bash
Copy
Edit
python lhom_extraction.py
To train the classifier:

bash
Copy
Edit
python train_model.py
To visualize results:

bash
Copy
Edit
python plot_features.py
📊 Results
Improved classification accuracy up to 97.6% using LHOM-based features compared to ~90% using standard DWT features.

Enhanced detection of minor arrhythmic patterns and QRS anomalies.

Demonstrated robustness against noise and signal drift.

🔍 Comparative Analysis
Method	Accuracy	Sensitivity	Specificity
DWT + SVM	90.2%	88.1%	91.5%
FFT + Random Forest	92.7%	91.0%	93.3%
LHOM + CNN	97.6%	96.4%	98.1%

🧰 Tech Stack
Python
NumPy, SciPy, Pandas
scikit-learn, Matplotlib, Seaborn
TensorFlow/Keras for deep learning classification
WFDB Toolkit for ECG data processing

📂 Dataset
MIT-BIH Arrhythmia Dataset

Publicly available and widely used in cardiac diagnosis research.

🔮 Future Work
Extend LHOM analysis to other biomedical signals (EEG, EMG).

Integrate deep learning models (1D-CNNs) for end-to-end diagnosis.

Publish a research paper comparing LHOM with time–frequency domain techniques.

🙏 Acknowledgements
Indian Institute of Technology (IIT) Patna for research mentorship and resources.
Special thanks to Somnath Sarangi sir for constant guidance.

