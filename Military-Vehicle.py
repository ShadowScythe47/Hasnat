import streamlit as st
from ultralytics import YOLOv10
from PIL import Image
import numpy as np
import json

# Load your trained models
model1 = YOLOv10('MilitaryData.pt')  # First model
model2 = YOLOv10('yolov10n.pt')  # Second model for comparison

# Streamlit interface
st.title("Fine-Tuning Comparison")
st.write("Upload an image to detect objects with both models:")

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    
    # Perform inference with both models
    results1 = model1(image_np, conf=0.25)
    results2 = model2(image_np, conf=0.25)
    
    # Process results from model 1
    if isinstance(results1, list):
        result1 = results1[0]  # Get the first result from the list
        annotated_image1 = result1.plot()  # Draw bounding boxes on the image
        annotated_image_pil1 = Image.fromarray(annotated_image1)  # Convert to PIL format

    # Process results from model 2
    if isinstance(results2, list):
        result2 = results2[0]  # Get the first result from the list
        annotated_image2 = result2.plot()  # Draw bounding boxes on the image
        annotated_image_pil2 = Image.fromarray(annotated_image2)  # Convert to PIL format

    # Display the original and annotated images from both models
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.image(annotated_image_pil1, caption='Fine-Tuned-Model', use_column_width=True)
    st.image(annotated_image_pil2, caption='Original-Model', use_column_width=True)

    # Save the annotated images (optional)
    annotated_image_pil1.save("annotated_image_model1.jpg")
    annotated_image_pil2.save("annotated_image_model2.jpg")

    # Get and display results in JSON format for both models
    json_results1 = result1.tojson()
    detections1 = json.loads(json_results1)
    st.write("Model 1 Detected objects:")
    for detection in detections1:
        label = detection['name']
        confidence = detection['confidence']
        box = detection['box']
        st.write(f"Label: {label}, Confidence: {confidence:.2f}, Box: {box}")

    json_results2 = result2.tojson()
    detections2 = json.loads(json_results2)
    st.write("Model 2 Detected objects:")
    for detection in detections2:
        label = detection['name']
        confidence = detection['confidence']
        box = detection['box']
        st.write(f"Label: {label}, Confidence: {confidence:.2f}, Box: {box}")
