#Retinal-Disease-Detection
#Retinal Disease Detection using ResNet50 + Attention Mechanism

---

Author: Vikas G J
B.Tech CSE (AIML) Presidency University, Bengaluru Project Duration: 2025

—

#Overview
This repository contains the complete implementation of a Retinal Disease Detection System using a hybrid ResNet50 + Attention Mechanism model. The project focuses on detecting retinal diseases from fundus images through deep learning–based feature extraction, attention enhancement, and multi-class classification.
The workflow includes:
* Data preprocessing
* Attention-based model architecture
* Training and validation
* Performance evaluation
* Visualization of results

—


Project Summary
Task 1: Data Preprocessing & Augmentation
* Objective: Prepare the retinal fundus images for deep learning-based classification.
* Steps Covered:
    * Loading of fundus image dataset
    * Image resizing, normalization, and label encoding
    * Train-validation-test splitting
    * Data augmentation techniques (rotation, zoom, flips)
    * Ensuring balanced input distribution

—


Task 2: ResNet50 + Attention Model Architecture
* Objective: Build an efficient deep learning model for retinal disease classification.
* Architecture Components:
    * ResNet50 pretrained backbone
    * Custom Attention Layer
    * Global Average Pooling
    * Dense layers for classification
    * Softmax output
* Model Configuration:
    * Optimizer: Adam
    * Loss: Categorical Crossentropy
    * Metrics: Accuracy, MAE

—

Task 3: Model Training & Validation
* Objective: Train the attention-based ResNet50 model on the processed dataset.
* Steps Covered:
    * Training with augmented data
    * Monitoring training & validation accuracy
    * Tracking model loss
    * Adjusting hyperparameters (epochs, batch size, learning rate)
* Outputs:
    * Accuracy plots
    * Loss plots

—


Task 4: Model Evaluation & Prediction Analysis
* Objective: Evaluate the model on unseen retinal images.
* Steps Covered:
    * Evaluation on test dataset
    * Metrics:
        * Accuracy
        * Loss
        * Mean Absolute Error (MAE)
    * Checking predictions on sample images
    * Understanding model behavior with Attention

—


Tools & Libraries Used
* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Matplotlib
* OpenCV
* Google Colab / Jupyter Notebook

File Structure

Retinal-Disease-Detection/
│
├── DL_projecttt.ipynb
├── data/
├── models/
├── results/
├── requirements.txt
└── README.txt

—

Contact
Vikas G J 📧 Email: vikasgjv@gmail.com 🌐 LinkedIn: linkedin.com/in/vikas-gj-979251296
