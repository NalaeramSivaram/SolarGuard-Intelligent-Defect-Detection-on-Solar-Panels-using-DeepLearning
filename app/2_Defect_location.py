import streamlit as st
from PIL import Image
import tempfile
import os
from ultralytics import YOLO
from visualize import draw_boxes_on_image

# Load YOLO model
model = YOLO("F:/DS/solar/Solardefect/data/runs/detect/train/weights/best.pt")

#Title image
Image1=Image.open("F:/DS/solar/banner-03.jpg")
st.image(Image1,caption="SolaeGuard",use_container_width=True)

# Load and display image in sidebar
image = Image.open("F:/DS/solar/wp8084814.jpg")  # Replace with your image path
st.sidebar.image(image, caption="SolarGuard", use_container_width=True)

# Application Details
st.sidebar.markdown("## About the App")
st.sidebar.markdown("""
This Streamlit application is designed for Solar Panel Defect Detection using  (YOLOv8)
✅ What it Does:

- Upload a solar panal image 
- Runs object detection using your trained YOLOv8 model (best.pt).
- 📊 Displays the image with bounding boxes and defect labels  
""")

st.title("🔆 SolarGuard: Solar Panel Defect Detection")
st.markdown("Upload an image of a solar panel to detect issues like dust, damage, or bird droppings.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        results = model(temp_file.name)

        st.subheader("📦 Detection Results")
        result_img = draw_boxes_on_image(image, results)
        st.image(result_img, caption="Detected Issues", use_column_width=True)

        st.info("Classes Detected:")
        names = model.names
        detected_classes = set()
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                detected_classes.add(names[cls_id])
        if detected_classes:
            for cls in detected_classes:
                st.write(f"✅ {cls}")
        else:
            st.write("No issues detected.")
