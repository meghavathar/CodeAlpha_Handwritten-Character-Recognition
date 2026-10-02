# ✍️ Handwritten Character Recognition

## 📌 Project Overview

Handwritten Character Recognition is a deep learning project that identifies handwritten characters from input images.

The project uses a **Convolutional Neural Network (CNN)** trained on the **EMNIST Balanced dataset**. The trained model can classify handwritten characters into 47 different classes.

A Streamlit web application is also provided where users can upload a handwritten character image and receive the predicted character along with its confidence score.

---

## 🎯 Objective

The main objective of this project is to develop a deep learning system capable of recognizing handwritten characters from images.

The project demonstrates:

* Image preprocessing
* Convolutional Neural Networks
* Deep learning model training
* Image classification
* Model evaluation
* Model saving and loading
* Streamlit deployment

---

## 📊 Dataset

This project uses the **EMNIST Balanced** dataset.

EMNIST is a collection of handwritten character images derived from the NIST Special Database 19.

Each image is:

* Grayscale
* 28 × 28 pixels
* Classified into one of 47 balanced character classes

### Dataset Sources

**NIST EMNIST Dataset:**
https://www.nist.gov/itl/products-and-services/emnist-dataset

**TensorFlow Datasets:**
https://www.tensorflow.org/datasets/catalog/emnist

The dataset is automatically downloaded through TensorFlow Datasets when `train_model.py` is executed.

The dataset does **not** need to be uploaded to GitHub.

---

## 🧠 Model

A Convolutional Neural Network (CNN) is used for handwritten character classification.

### CNN Architecture

```text
Input Image
     ↓
28 × 28 × 1
     ↓
Convolutional Layer
     ↓
Max Pooling
     ↓
Convolutional Layer
     ↓
Max Pooling
     ↓
Convolutional Layer
     ↓
Flatten
     ↓
Dense Layer
     ↓
Dropout
     ↓
Softmax
     ↓
47 Character Classes
```

---

## 🔄 Project Workflow

```text
EMNIST Dataset
       ↓
Image Preprocessing
       ↓
Normalization
       ↓
CNN Model
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Save Trained Model
       ↓
Streamlit Application
       ↓
Upload Handwritten Character
       ↓
Prediction
```

---

## ✨ Features

* Recognizes handwritten characters
* CNN-based deep learning model
* EMNIST Balanced dataset
* Image preprocessing
* Character classification
* Prediction confidence
* Top 5 predictions
* Interactive Streamlit interface
* Automatic dataset downloading

---

## 🛠️ Technologies Used

* Python
* TensorFlow
* TensorFlow Datasets
* NumPy
* Pillow
* Streamlit
* Convolutional Neural Network
* Deep Learning

---

## 📁 Project Structure

```text
Handwritten_Character_Recognition/
│
├── train_model.py
├── app.py
├── requirements.txt
├── README.md
│
├── handwritten_character_model.keras
└── labels.json
```

### File Description

| File                                | Description                                     |
| ----------------------------------- | ----------------------------------------------- |
| `train_model.py`                    | Downloads the EMNIST dataset and trains the CNN |
| `app.py`                            | Streamlit web application                       |
| `requirements.txt`                  | Required Python libraries                       |
| `README.md`                         | Project documentation                           |
| `handwritten_character_model.keras` | Trained CNN model                               |
| `labels.json`                       | Character class labels                          |

---

## ⚙️ Installation

Make sure Python is installed.

Install the required libraries:

```bash
pip install -r requirements.txt
```

If you are using Python 3.14, you can also use:

```bash
py -3.14 -m pip install -r requirements.txt
```

---

## 🚀 Train the Model

Run:

```bash
py -3.14 train_model.py
```

The EMNIST Balanced dataset will be downloaded automatically.

After training is completed, the following files will be created:

```text
handwritten_character_model.keras
labels.json
```

---

## 🌐 Run the Streamlit Application

After the model has been trained, run:

```bash
py -3.14 -m streamlit run app.py
```

Streamlit will provide a local URL similar to:

```text
http://localhost:8501
```

Open the URL in your browser.

---

## 🖼️ How to Use the Application

1. Start the Streamlit application.
2. Upload an image containing a handwritten character.
3. The image is converted to grayscale.
4. The image is resized to 28 × 28 pixels.
5. Pixel values are normalized.
6. The processed image is passed to the CNN.
7. The CNN predicts the character.
8. The application displays the predicted character and confidence.
9. The application also displays the top 5 predictions.

---

## 📈 Prediction Output

The application displays:

```text
Predicted Character: A

Confidence: 92.35%
```

It also displays the top 5 predictions and their probabilities.

---

## 🔮 Future Improvements

The project can be extended by adding:

* Drawing canvas for handwritten characters
* Recognition of complete words
* Recognition of handwritten sentences
* Optical Character Recognition (OCR)
* CRNN-based sequence modeling
* Better image preprocessing
* Real-time character recognition
* Web deployment
* Mobile application

---

## 👩‍💻 Author

**Megha Vathar**

### Project

**Handwritten Character Recognition using CNN and EMNIST**

---

## ⭐ Conclusion

This project demonstrates how deep learning and convolutional neural networks can be used to recognize handwritten characters from images.

It provides a complete workflow from dataset loading and image preprocessing to CNN training, prediction, and deployment using Streamlit.
