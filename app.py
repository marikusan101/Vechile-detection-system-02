
import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import os
import torch
import tempfile

# Set page title and favicon
st.set_page_config(page_title="YOLOv5 Traffic Object Detection", page_icon="🚗")

st.title("🚗 YOLOv5 Traffic Object Detection")
st.write("Upload an image and let the YOLOv5 model detect traffic objects.")

# Load the YOLOv5 model
@st.cache_resource # Cache the model loading to avoid reloading on every rerun
def load_model():
    try:
        model = YOLO('yolov5_traffic_detection_best.pt') # Model weights are in the same directory
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Load class names
@st.cache_data # Cache class names
def load_class_names(file_path):
    try:
        with open(file_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        return class_names
    except Exception as e:
        st.error(f"Error loading class names: {e}")
        return ['Unknown']

class_names_path = 'classes.txt' # Class names are in the same directory
class_names = load_class_names(class_names_path)

if model:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Detecting...")

        # Perform inference
        try:
            # Save the uploaded image to a temporary file for YOLO to read
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                image.save(tmp_file.name)
                tmp_file_path = tmp_file.name

            results = model.predict(source=tmp_file_path, conf=0.25) # Adjust confidence threshold if needed

            # Process and display results
            for r in results:
                im_array = r.plot()  # plot a BGR numpy array of predictions
                im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
                st.image(im, caption='Detection Results', use_column_width=True)

                # Optionally display detected objects with confidence
                if len(r.boxes) > 0:
                    st.subheader("Detected Objects:")
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        label = class_names[cls] if cls < len(class_names) else f"Class {cls}"
                        st.write(f"- {label} (Confidence: {conf:.2f})")
                else:
                    st.write("No objects detected.")

            # Clean up temporary file
            os.unlink(tmp_file_path)

        except Exception as e:
            st.error(f"Error during detection: {e}")
            if 'CUDA out of memory' in str(e):
                st.warning("Consider running on a smaller image or reducing batch size if possible.")

st.markdown("---")
st.info("This app uses a YOLOv5 nano model trained on the Traffic Road Object Detection dataset.")
