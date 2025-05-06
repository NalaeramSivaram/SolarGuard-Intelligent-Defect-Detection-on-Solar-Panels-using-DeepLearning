import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
import numpy as np
from PIL import Image

''' Set page config
st.set_page_config(page_title="Solar Panel Condition Classifier", layout="centered")'''

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
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array



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
    st.image(img, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Classifying..."):
        processed_img = preprocess_image(img)
        prediction = model.predict(processed_img)
        predicted_class = class_names[np.argmax(prediction)]

     # Get top 2 predictions
    top_indices = prediction[0].argsort()[-2:][::-1]  # indices of top 2 values
    top_classes = [(class_names[i], prediction[0][i]) for i in top_indices]

    # Display top prediction
    st.success(f"🧠 Top Prediction: **{top_classes[0][0]}** ({top_classes[0][1]:.2%})")

    # Display second-best prediction
    st.info(f"🔍 Second Likely: **{top_classes[1][0]}** ({top_classes[1][1]:.2%})")