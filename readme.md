# 👕 Apparel Image Classification App

This project is an **image classification application** that classifies apparel images into 10 categories using a **Convolutional Neural Network (CNN)** built with **TensorFlow/Keras**. The app includes a **Gradio web interface** with a login system for interactive predictions.

---

## **Features**

- CNN model for classifying apparel images.
- Supports **10 apparel categories**.
- **Gradio UI** for uploading images and getting predictions.
- Login system for secure access.
- Displays **prediction probabilities** for all classes.
- Normalizes and preprocesses images automatically.

---

## **Folder Structure**


---

## **Requirements / Libraries**

- Python 3.x  
- TensorFlow  
- Gradio  
- Matplotlib  
- Numpy  

Install dependencies with:

```bash
pip install tensorflow gradio matplotlib numpy
CNN Model Architecture

Conv2D (64 filters, 3x3, ReLU) → MaxPooling2D

Conv2D (128 filters, 3x3, ReLU) → MaxPooling2D

Conv2D (256 filters, 3x3, ReLU) → MaxPooling2D

Flatten → Dense(512, ReLU) → Dropout(0.5) → Dense(10, Softmax)

Optimizer: Adam

Loss: Sparse Categorical Crossentropy

Metrics: Accuracy

Login credentials:

Username: admin

Password: 1234