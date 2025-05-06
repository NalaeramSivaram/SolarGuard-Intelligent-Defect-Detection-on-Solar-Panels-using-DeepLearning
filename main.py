import streamlit as st
import importlib.util
import os
from PIL import Image

st.set_page_config(page_title="SolarGuard", layout="wide")
st.sidebar.title("☀️🛡️ SolarGuard Menu")

#Title image
Image1=Image.open("F:/DS/solar/banner-03.jpg")
st.image(Image1,caption="SolaeGuard",use_container_width=True)

# Load and display image in sidebar
image = Image.open("F:/DS/solar/wp8084814.jpg")  # Replace with your image path
st.sidebar.image(image, caption="SolarGuard", use_container_width=True)

page = st.sidebar.radio("Go to", ["Home", "Classification", "Object Detection"])

if page == "Home":
    st.title("🔆 SolarGuard")
    st.markdown("Welcome to the Solar Panel Defect Detection System.")

# Application Details
st.markdown("""
            Solar energy is a crucial renewable resource, but the accumulation of **dust**, **snow**, **bird droppings**, and **physical/electrical damage** on solar panels reduces their efficiency. While manual monitoring is time-consuming and expensive, **automated detection** can help improve efficiency and reduce maintenance costs.
This **Streamlit app** used for both classification and object detection to accurately identify and localize different types of obstructions or damages on solar panels. 
           
#### 🎯 **The objective**
-**Classify** solar panel images into six categories: 
-Clean 
-Dusty 
-Bird-Drop 
-Electrical-Damage 
-Physical-Damage 
-Snow-Covered

-**Detect and localize** the presence of:
-Dust 
-Bird droppings 
-Electrical-Damage 
-Physical-Damage
""")

# Utility function to run external .py files
def run_script(script_path):
    spec = importlib.util.spec_from_file_location("module.name", script_path)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)

# Run external apps
if page == "Classification":
    run_script(os.path.join("classification_app", "classify.py"))

elif page == "Object Detection":
    run_script(os.path.join("detection_app", "detect.py"))