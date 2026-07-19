# Cocoa Disease Detection using Deep Learning

## Project Overview
This project aims to detect and classify diseases in cacao leaves using deep learning techniques. It explores both a custom Convolutional Neural Network (CNN) and a Transfer Learning approach using a pre-trained MobileNetV2 model. A Streamlit web application is also provided for interactive inference.

## Dataset

The dataset used for this project is `serranosebas/enfermedades-cacao-yolov4` from Kaggle. It was downloaded using `kagglehub.dataset_download`. The dataset contains images of cacao leaves classified into three categories:
- **Fito** (Phytophthora)
- **Monilia** (Moniliophthora Roreri)
- **Sana** (Healthy)

Upon exploration, the dataset structure was found to be directly composed of class folders, without explicit `train`/`validation`/`test` splits. Image counts per class were approximately 100-107 images, totaling 312 images for training and validation after an 80/20 split.

## Setup and Installation

To set up the environment and run this project, follow these steps:

1.  **Clone the repository (if applicable) or ensure all project files are in one directory.**
2.  **Install Python dependencies:** A `requirement` file (or `requirements.txt`) containing all necessary libraries and their versions has been generated. Install them using pip:
    ```bash
    pip install -r requirement
    ```
    _Note: Ensure you have `pip` installed and correctly configured._

## Model Development

### 1. Custom Convolutional Neural Network (CNN)
A simple sequential CNN model was initially built with `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`, and `Dropout` layers. The model was compiled with `adam` optimizer and `categorical_crossentropy` loss.

**Performance:** This model showed severe underfitting, with both training and validation accuracy hovering around 30-40% and high, flat loss curves. The classification report indicated that the model primarily predicted a single class ('Sana') regardless of the true label.

### 2. Transfer Learning with MobileNetV2
Given the poor performance of the custom CNN, a transfer learning approach was adopted using the MobileNetV2 model, pre-trained on ImageNet. The base MobileNetV2 model's layers were frozen, and a custom classification head (with `GlobalAveragePooling2D`, `Dense`, and `Dropout` layers) was added.

**Performance:** This model, named `model_tl`, showed improved validation accuracy (around 45.16%) and a lower loss (1.1049) compared to the simple CNN. However, the performance is still problematic, with significant confusion between classes (as shown in the confusion matrix).

**Model Export:** The trained MobileNetV2 transfer learning model is saved as `cacao_disease_mobilenetv2_finetuned_model.keras`.

## Streamlit Web Application
A Streamlit-based web application (`app.py`) has been created to provide an interactive interface for predicting cacao leaf diseases using the exported `cacao_disease_mobilenetv2_finetuned_model.keras`.

### How to Run the Web App

1.  **Ensure `app.py` and `cacao_disease_mobilenetv2_finetuned_model.keras` are in the same directory.**
2.  **Open your terminal or command prompt.**
3.  **Navigate to the directory** where you saved `app.py`.
4.  **Run the app** using the Streamlit command:
    ```bash
    streamlit run app.py
    ```
5.  This will open the application in your web browser, allowing you to upload images and get predictions.

## Future Work and Improvements

-   **Fine-tuning the MobileNetV2 base layers:** Unfreeze some of the top layers of the MobileNetV2 base model and retrain with a very low learning rate for further performance improvement.
-   **Data Augmentation Strategy:** Experiment with more diverse or tailored data augmentation techniques.
-   **Dataset Expansion:** Obtain a larger and more balanced dataset with more images per class.
-   **Other Transfer Learning Models:** Explore other state-of-the-art pre-trained models (e.g., EfficientNet, ResNet).
-   **Hyperparameter Tuning:** Systematically tune hyperparameters for both the training process and model architecture.
