import streamlit as st
import torch
import cv2
import pandas as pd
import mediapipe as mp
import os
import numpy as np

# Load the pre-trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Streamlit app title
st.title("Object Detection and Back Support Analysis")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    img = np.array(cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Perform object detection
    results = model(img)

    # Get the detection results (bounding boxes and labels)
    detections = results.pandas().xyxy[0]

    # Filter objects of interest: Screens, Laptops, Keyboard, Mouse
    objects_of_interest = {
        'Screens': ['tv', 'monitor'],
        'Laptops': ['laptop'],
        'Keyboard and Mouse': ['keyboard', 'mouse']
    }

    # Count detected objects
    object_counts = {category: 0 for category in objects_of_interest.keys()}

    for category, labels in objects_of_interest.items():
        count = detections[detections['name'].isin(labels)].shape[0]
        object_counts[category] += count

    # Convert the object counts to a table using pandas
    df_counts = pd.DataFrame(list(object_counts.items()), columns=['Object', 'Count'])

    # Display the detected objects
    st.subheader("Detected Objects:")
    st.write(df_counts)

    # Perform pose estimation
    results_pose = pose.process(img_rgb)

    # Extract keypoints for the upper, mid, and lower back
    if results_pose.pose_landmarks:
        landmarks = results_pose.pose_landmarks.landmark

        # Upper back (shoulders)
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # Mid back (spine or mid-point between shoulders)
        mid_back_x = (left_shoulder.x + right_shoulder.x) / 2

        # Lower back (hips)
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

        # Define thresholds for detecting support
        chair_threshold = 0.8  # Assume chair is at the rightmost 20% of the image

        def check_back_support(keypoint_x):
            """Check if the keypoint is near the back of the chair."""
            return "Supported" if keypoint_x >= chair_threshold else "Not Supported"

        # Check support for each section
        upper_back_support = check_back_support((left_shoulder.x + right_shoulder.x) / 2)
        mid_back_support = check_back_support(mid_back_x)
        lower_back_support = check_back_support((left_hip.x + right_hip.x) / 2)

        # Output the back support analysis
        st.subheader("Back Support Analysis:")
        st.write(f"Upper back: {upper_back_support}")
        st.write(f"Mid back: {mid_back_support}")
        st.write(f"Lower back: {lower_back_support}")

    else:
        st.write("No pose detected.")

    # Optional: Display the image with detections
    st.image(results.render()[0], caption='Processed Image', use_column_width=True)
