# SolarGuard-Intelligent-Defect-Detection-on-Solar-Panels-using-DeepLearning
This project aims to develop machine learning models for both classification and object detection to accurately identify and localize different types of obstructions or damages on solar panels. 
**The objective is to:**
**I.Classify solar panel** images into six categories: Clean, Dusty, Bird-Drop, Electrical-Damage, Physical-Damage, and Snow-Covered.
**II.Detect and localize** the presence of dust, bird droppings, or damages on the panel using object detection models.

 
 Step-by-Step Procedure
**1. Problem Definition**
Goal: Classify solar panel images into one of six conditions:
Clean, Dusty, Bird-Drop, Electrical-Damage, Physical-Damage, Snow-Covered.

**2.Detect and Localizing** the presence of dust, bird droppings, or damages on the panel using object detection models.
Organize the images in folders named after their respective classes.

dataset/
├── Clean/
├── Dusty/
├── Bird-Drop/
├── Electrical-Damage/
├── Physical-Damage/
└── Snow-Covered/
**3. Data Preprocessing**
Resize images to a standard size (e.g., 224x224).
Normalize pixel values (0–1) by dividing by 255.
Apply data augmentation to increase dataset diversity (e.g., rotation, zoom, flipping).
Split dataset into:
Training set (e.g., 80%)
Validation set (e.g., 20%)

**4. Handle Class Imbalance**
Use class_weight to penalize under-represented classes.
Visualize class distribution to confirm imbalance before training.

**5. Model Selection: MobileNetV2**
Use MobileNetV2 as the base model with pretrained weights (imagenet).
Set include_top=False to customize the output layer.
Freeze base layers initially (to avoid damaging pretrained features).

**6. Add Custom Classification Head**
Add layers on top of MobileNetV2:
Global Average Pooling
Dense Layer with ReLU
Dropout Layer (to reduce overfitting)
Final Dense Layer with Softmax (for classification into 6 categories)

**7. Compile the Model**
Optimizer: Adam with learning rate 1e-4
Loss: categorical_crossentropy
Metrics: accuracy

**8. Train the Model**
Train using model.fit() with:
train_generator and val_generator
class_weight to address imbalance
EarlyStopping and ReduceLROnPlateau callbacks

**9. Fine-Tuning**
Unfreeze top layers of MobileNetV2 (from a certain layer onward).
Recompile the model with a lower learning rate.
Train again for a few epochs (fine_tune_epochs) to improve validation accuracy.

**10. Evaluate and Save the Model**
Evaluate performance using validation data (accuracy, loss).
Save final model using model.save('filename.h5').


**Object Detection (Optional):** Localize and detect:
  - Dust  
  - Bird Droppings  
  - Electrical Damage  
  - Physical Damage

---

## 🧠 Approach

### 1. Data Preprocessing & Annotation
- Applied image augmentation for dataset balancing.
- Resized images to compatible dimensions (e.g., 224×224).
- Annotated images with bounding boxes (for detection).
- Normalized pixel values to improve training stability.

### 2. Model Training
- **Classification:** Trained CNN models (ResNet, EfficientNet, MobileNet).
- **Object Detection:** Used YOLOv8 and Faster R-CNN for precise localization.

### 3. Model Evaluation
- **Classification Metrics:** Accuracy, Precision, Recall, F1-Score.
- **Object Detection Metrics:** 
  - mAP (Mean Average Precision)  
  - IoU (Intersection over Union)

---

## 🖥️ Streamlit Web App

### 🔍 Features
- Upload solar panel images.
- Get predictions with confidence scores.
- View object detection results with bounding boxes (optional).
- Simple, interactive UI for non-technical users.

 Deployment (Optional)
Build a Streamlit with multi-page app.

Load and test model with real user-uploaded solar panel images.
