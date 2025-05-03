import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Set page config
st.set_page_config(page_title="Solar Panel Condition Classifier", layout="centered")

# Load model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("solar_panel_classifier_correcting.h5")

model = load_model()

# Class names (must match training order)
class_names = ['Bird-Drop', 'Clean', 'Dusty', 'Electrical-Damage', 'Physical-Damage', 'Snow-Covered']

# Preprocessing function
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array


#Title image
Image1=Image.open("F:/DS/solar/banner-03.jpg")
st.image(Image1,caption="SolaeGuard",use_container_width=True)

# Load and display image in sidebar
image = Image.open("F:/DS/solar/wp8084814.jpg")  # Replace with your image path
st.sidebar.image(image, caption="SolarGuard", use_container_width=True)

# Application Details
st.sidebar.markdown("## About the App")
st.sidebar.markdown("""
This Streamlit application is designed for intelligent solar panel inspection.

- 🧠 Uses deep learning for defect classification  
- 📷 Supports image upload  
- 📊 Displays prediction results and confidence  
""")

# Streamlit UI
st.title("🔎 SolarGuard - Panel Condition Classifier")
st.write("Upload a solar panel image to detect its condition.")

uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Classifying..."):
        processed_img = preprocess_image(img)
        prediction = model.predict(processed_img)
        predicted_class = class_names[np.argmax(prediction)]

    st.success(f"🧠 Predicted Condition: **{predicted_class}**")

    # Show probabilities
    st.subheader("Prediction Confidence:")
    for i, prob in enumerate(prediction[0]):
        st.write(f"{class_names[i]}: {prob:.2%}")