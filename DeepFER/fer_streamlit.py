import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import cv2

# Load the saved model (.keras or .h5)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('fer_model.keras')  # Change filename if using .h5
    return model

model = load_model()

# Emotion labels as per your class indices
class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# UI
st.title("😊 Facial Emotion Recognition")
st.write("Upload a **grayscale face image** (48×48 pixels), and the model will predict the emotion.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file).convert("L")  # Grayscale
    image_resized = image.resize((48, 48))
    img_array = np.array(image_resized)

    # Show uploaded image
    st.image(image_resized, caption='Uploaded Image', width=150)
    
    # Normalize and reshape
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 48, 48, 1)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class] * 100

    st.markdown(f"### 🎯 Predicted Emotion: **{class_labels[predicted_class].capitalize()}**")
    st.write(f"Confidence: `{confidence:.2f}%`")
