import streamlit as st
from PIL import Image
import tempfile
import numpy as np
import cv2
from ultralytics import YOLO

st.title("🎯 Defect Localization with YOLOv8")

model = YOLO("F:/DS/solar/Solardefect/data/runs/detect/train/weights/best.pt")

def draw_boxes_on_image(image, results):
    image_np = np.array(image)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = result.names[cls_id]
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_np, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return Image.fromarray(image_np)

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        results = model(temp_file.name)

        result_img = draw_boxes_on_image(image, results)
        st.image(result_img, caption="Detected Issues")

        names = model.names
        detected_classes = set()
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                detected_classes.add(names[cls_id])
        if detected_classes:
            st.info("Classes Detected:")
            for cls in detected_classes:
                st.write(f"✅ {cls}")
        else:
            st.write("No issues detected.")
