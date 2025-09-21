import gradio as gr
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------------
# Dataset Setup
# ---------------------------
data_dir = r"C:\Users\Manoj\Downloads\archive\Apparel images dataset new"
img_size = (128, 128)
batch_size = 32

# Load datasets (raw, before mapping)
raw_train_data = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

raw_validation_data = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

# Get class names before mapping
class_names = raw_train_data.class_names

# Normalize datasets
norm = layers.Rescaling(1./255)
train_data = raw_train_data.map(lambda x, y: (norm(x), y)).prefetch(tf.data.AUTOTUNE)
validation_data = raw_validation_data.map(lambda x, y: (norm(x), y)).prefetch(tf.data.AUTOTUNE)

# ---------------------------
# Model Definition
# ---------------------------
model = models.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ---------------------------
# Train Model
# ---------------------------
history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=5
)

# Evaluate
test_loss, test_acc = model.evaluate(validation_data)
print("✅ Validation Accuracy:", test_acc)

# ---------------------------
# Prediction Function
# ---------------------------
def predict_image(img):
    img = tf.image.resize(img, img_size) / 255.0
    img = np.expand_dims(img, axis=0)  # add batch dimension
    preds = model.predict(img)[0]
    top_idx = np.argmax(preds)
    confidence = preds[top_idx]
    return {class_names[i]: float(preds[i]) for i in range(len(class_names))}, f"✅ Predicted: {class_names[top_idx]} ({confidence:.2f})"

# ---------------------------
# Login Validation
# ---------------------------
def login(username, password):
    if username == "admin" and password == "1234":
        return gr.update(visible=False), gr.update(visible=True)
    else:
        return gr.update(value="❌ Invalid login!"), gr.update(visible=False)

# ---------------------------
# Gradio UI
# ---------------------------
with gr.Blocks() as demo:
    # Page 1: Login
    with gr.Row(visible=True) as login_page:
        with gr.Column():
            gr.Markdown("## 🔑 Login to Access Apparel Classification App")
            username = gr.Textbox(label="Username")
            password = gr.Textbox(label="Password", type="password")
            login_btn = gr.Button("Login")
            login_msg = gr.Textbox(label="Login Status")

    # Page 2: Image Classification App
    with gr.Row(visible=False) as app_page:
        with gr.Column():
            gr.Markdown("## 👕 Apparel Image Classification")
            image_input = gr.Image(type="numpy", label="Upload Apparel Image")
            predict_btn = gr.Button("Predict")
            output_label = gr.Label(num_top_classes=3, label="Prediction")
            output_text = gr.Textbox(label="Result")

    # Button Actions
    login_btn.click(fn=login, inputs=[username, password], outputs=[login_msg, app_page])
    predict_btn.click(fn=predict_image, inputs=image_input, outputs=[output_label, output_text])

# ---------------------------
# Launch
# ---------------------------
if __name__ == "__main__":
    demo.launch()
