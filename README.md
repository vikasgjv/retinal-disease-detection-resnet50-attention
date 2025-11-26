Retinal-Disease-Detection
Retinal Disease Detection using ResNet50 + Attention
Author: Vikas G J

B.Tech CSE (AIML)
Presidency University, Bengaluru
Project Period: 2025

Overview
This repository contains the complete implementation of a Retinal Disease Detection System built using ResNet50 with an Attention Mechanism. The project focuses on analyzing retinal fundus images to classify different retinal conditions with improved accuracy and interpretability.
The model includes preprocessing, feature extraction, attention-based enhancement, training, evaluation, and visualization of performance metrics. This work is designed for academic learning, medical image analysis research, and deep learning experimentation.

Project Summary
1️⃣ Data Preprocessing & Preparation
Imported and cleaned retinal fundus images
Resized and normalized image data
Converted labels into model-readable format
Split dataset into training, validation, and testing sets
Applied augmentations to improve generalization (rotation, flip, zoom)

2️⃣ Model Architecture: ResNet50 + Attention
ResNet50 used as base feature extractor
Added custom Attention Block to focus on disease-specific retinal regions
Fully connected layers for classification
Softmax output for multi-class prediction
Compiled with:
Optimizer: Adam
Loss: Categorical Crossentropy
Metrics: Accuracy, MAE

3️⃣ Training Process
Trained the model on cleaned and augmented image dataset
Visualized:
Training & validation accuracy
Training & validation loss
Tuned hyperparameters such as batch size, epochs, and learning rate
Ensured stable training using callbacks where needed

4️⃣ Model Evaluation
Evaluated the model on unseen test data using:
Accuracy
Loss
Mean Absolute Error (MAE)
Confusion Matrix (if implemented)
Classification performance across multiple retinal categories

5️⃣ Output & Interpretation
The model predicts the retinal disease class for each input fundus image
Attention mechanism helps in focusing on disease-related regions
Visualization graphs show training progress and performance trends

Tools & Libraries Used
Python
TensorFlow / Keras
NumPy, Pandas
Matplotlib
OpenCV
Google Colab / Jupyter Notebook

File Structure
Retinal-Disease-Detection/
DL_projecttt.ipynb/
README.md

Contact

Vikas G J
📧 Email: vikasgjv@gmail.com
🌐 LinkedIn: linkedin.com/in/vikas-gj-979251296
