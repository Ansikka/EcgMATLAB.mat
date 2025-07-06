# LHOM-Based ECG Signal Classification

This project focuses on classifying ECG signals (Normal vs Diseased) using **Local Higher Order Moments (LHOM)** as statistical features. It combines **signal processing**, **feature engineering**, and **machine learning (SVM, KNN, Random Forest)** in MATLAB.

## Project Objectives
- Simulate or load ECG signals (normal & diseased)
- Extract Local Higher Order Moments (up to 100th order)
- Normalize and log-transform moment features
- Save features to CSV for analysis
- Train ML models (SVM, KNN, Random Forest)
- Analyze performance and overfitting behavior

##  Folder Structure
lhom-ecg-classification

├──    ecg_normal1.mat # Sample normal ECG signal
├──    ecg_disease1.mat # Sample diseased ECG signal
├──    extract_lhom_features.m # Main script for feature extraction
├──    lhom_features_10000.csv # Saved 10000 samples with LHOM features and labels
├──    classify_models.m # SVM, KNN, Random Forest classifiers
├──    overfitting_plot.m # Overfitting detection and plotting
└──    README.md # Project documentation

##  Features Implemented

###  1. ECG Signal Preprocessing
- Loading `.mat` files of ECG signals
- Signal repetition to generate 5000 samples/class (10,000 total)
- Z-score normalization of each segment

###  2. LHOM Feature Extraction
- Moments computed from 1st to 100th order
- Logarithmic transformation using `log10(abs(moment) + eps)`
- Separate LHOM profile for normal vs diseased ECG

###  3. CSV Export
- 100 LHOM features per sample
- Final dataset: `10000 x 101` (last column = label)
- Saved as `lhom_features_10000.csv`

###  4. Machine Learning Classification
- **SVM** with RBF kernel  
- **KNN** with k = 5  
- **Random Forest** with 100 trees  
- Train-test split (70/30)
- Accuracy and confusion matrix for each

###  5. Overfitting Visualization
- Varying training sizes: 10% to 90%
- Accuracy plots: Train vs Test
- Helps detect overfitting behavior for all models

##  6. Sample Output
   SVM Accuracy : 96.20%
   KNN Accuracy : 94.85%
   Random Forest Acc : 97.40%

###  7. Overfitting Plot
- SVM, KNN, and RF accuracies shown against increasing train sizes
- Helps evaluate model robustness and generalization

##  8. Requirements
- MATLAB R2021a or later
- Signal Processing Toolbox (optional but helpful)

##  9. Future Scope
- Apply PCA or t-SNE on LHOM features for visualization
- Use raw ECG input with CNN or RNN models for temporal classification
- Extend dataset with real ECG samples (e.g., MIT-BIH)
- Explore hybrid feature sets: LHOM + Wavelet + Entropy

##  10. Citation
If this project helped you, please consider citing it or starring the repository ⭐.
