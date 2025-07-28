# Multi-Task-Learning-for-Facial-Attribute-and-Landmark-Prediction-on-CelebA

This project explores the application of multi-task learning (MTL) for simultaneously predicting facial attributes and localizing facial landmarks from images in the CelebA dataset. The implementation uses PyTorch and a ResNet-18 backbone.

-----

## 📖 Table of Contents

  * [Features](https://www.google.com/search?q=%23-features)
  * [Dataset](https://www.google.com/search?q=%23-dataset)
  * [Methodology](https://www.google.com/search?q=%23-methodology)
      * [Exploratory Data Analysis](https://www.google.com/search?q=%23-exploratory-data-analysis)
      * [Data Management](https://www.google.com/search?q=%23-data-management)
      * [Model Architecture](https://www.google.com/search?q=%23-model-architecture)
  * [Training](https://www.google.com/search?q=%23-training)
  * [Results](https://www.google.com/search?q=%23-results)
  * [Usage](https://www.google.com/search?q=%23-usage)
  * [Dependencies](https://www.google.com/search?q=%23-dependencies)

-----

## ✨ Features

  - **Multi-Task Learning:** A single model is trained to perform two different tasks: facial attribute classification and landmark regression.
  - **Single-Task Baseline:** For comparison, two separate single-task models are also trained for each individual task.
  - **Pre-trained Backbone:** Utilizes a ResNet-18 model pre-trained on ImageNet, with its final layer adapted for our specific tasks.
  - **Data Augmentation:** Applies standard image transformations (resizing, normalization) to the input images.
  - **Early Stopping:** Implements an early stopping mechanism during training to prevent overfitting and save the best model based on validation loss.
  - **Comprehensive Evaluation:** The models are evaluated using a variety of metrics:
      - **Attributes:** Macro Accuracy, Macro Precision, Macro Recall, and Macro F1-Score.
      - **Landmarks:** Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).
  - **Visualization:** Includes code to visualize the model's predictions on sample images, showing both the predicted landmarks and attributes against the ground truth.

-----

## 📊 Dataset

The project uses the **CelebA (CelebFaces Attributes) dataset**, which is a large-scale face attributes dataset with over 200,000 celebrity images. Each image has annotations for 40 facial attributes and 5 landmark locations. For this project, we focus on three attributes:

  - High Cheekbones
  - Mouth Slightly Open
  - Smiling

The dataset is downloaded using the `kagglehub` library.

-----

## 🛠️ Methodology

### Exploratory Data Analysis

The notebook begins with an EDA to understand the relationships between facial attributes and landmarks. A correlation heatmap is generated, revealing interesting connections. For example, the 'Smiling' attribute shows a strong correlation with the x-coordinates of the mouth corners.

### Data Management

A custom PyTorch `Dataset` class, `CelebAMultiTaskDataset`, is implemented to efficiently load and preprocess the images, attributes, and landmark data. The dataset is divided into training, validation, and testing partitions as provided in the original CelebA dataset.

### Model Architecture

The core of the project is a **Multi-Task Learning model** (`MultiTaskCelebAModel`) with a shared ResNet-18 backbone. The final fully connected layer of the ResNet-18 is replaced with two task-specific heads:

1.  **Attribute Head:** A sequential model with a linear layer followed by a ReLU activation and another linear layer to output the logits for the three attributes.
2.  **Landmark Head:** A similar sequential model that outputs the 10 landmark coordinates (5 points with x and y).

For comparison, two **Single-Task models** (`AttrTaskCelebAModel` and `LandmarkTaskCelebAModel`) are also created, each with its own ResNet-18 backbone and a single head for its respective task.

-----

## 🚀 Training

  - The models are trained for 15 epochs with a learning rate of 0.001.
  - **Loss Functions:**
      - **Attribute Task:** Binary Cross-Entropy with Logits Loss (`BCEWithLogitsLoss`).
      - **Landmark Task:** Mean Squared Error Loss (`MSELoss`).
  - For the MTL model, the total loss is a weighted sum of the attribute and landmark losses, with weights of 50.0 and 100.0, respectively.
  - An **Adam optimizer** is used for training.
  - **Early stopping** with a patience of 2 is used to monitor the validation loss and stop training if there's no improvement.

-----

## 📈 Results

After training, the models are evaluated on the test set, yielding the following results:

### Multi-Task Learning Model

  - **Attribute Recognition:**
      - Macro Accuracy: 0.7800
      - Macro Precision: 0.7914
      - Macro Recall: 0.7643
      - Macro F1-Score: 0.7776
  - **Landmark Localization:**
      - Mean Absolute Error (MAE): 0.0128
      - Mean Squared Error (MSE): 0.0003
      - Root Mean Squared Error (RMSE): 0.0180

### Single-Task Models

  - **Attribute Recognition:**
      - Macro Accuracy: 0.7694
      - Macro Precision: 0.8388
      - Macro Recall: 0.6717
      - Macro F1-Score: 0.7456
  - **Landmark Localization:**
      - Mean Absolute Error (MAE): 0.0137
      - Mean Squared Error (MSE): 0.0004
      - Root Mean Squared Error (RMSE): 0.0189

The results show that the **multi-task learning model performs better** on both tasks compared to the single-task models, demonstrating the benefit of learning shared representations.

-----

## 💻 Usage

To run this project, you will need to have a Kaggle account and have your API credentials set up. The notebook will automatically download the dataset. Simply run the cells in the `MTL_celebA_final.ipynb` notebook in sequential order.

-----

## 📦 Dependencies

  - PyTorch
  - KaggleHub
  - Pandas
  - Seaborn
  - Matplotlib
  - scikit-learn
  - Pillow (PIL)
