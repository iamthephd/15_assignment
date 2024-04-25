import streamlit as st
import joblib
import numpy as np
from PIL import Image

import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

from run import get_landmarks, get_process_landmarks, get_features



class_dict = {1: 'Angry', 2: 'Disgust', 3: 'Fear', 4: 'Happy', 5: 'Neutral', 6: 'Sad', 7: 'Surprise'}


# Model Deployment
# function to load the data
def get_img_data(image):
    # Checking if the image is valid and has content
    if image is None:
        raise ValueError("No image data received.")
    
    if isinstance(image, Image.Image):
        # Convert PIL Image to numpy array
        image_np = np.array(image)
    else:
        # Assume it is already a numpy array
        image_np = image
    
    # Check the shape of the numpy array
    if image_np.ndim != 3 or image_np.shape[2] != 3:
        raise ValueError("Image must be a 3-channel (RGB) image.")
    
    detection_result = get_landmarks(image)
    landmarks_processed = get_process_landmarks(detection_result)
    features = get_features(landmarks_processed)
    return detection_result, features


# function to plot the facial landmarks
def draw_landmarks_on_image(img, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(img)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image

def load_model():
    try:
        return joblib.load('emotion_recognition_model.pkl')
    except Exception as e:
        print("Failed to load model:", e)
        return None


def load_model():
    return joblib.load('emotion_recognition_model.pkl')

model = load_model()

def predict(image, model=model):
    # Loading model
    detection_result, features = get_img_data(image)
    prediction = model.predict_proba(features.reshape(1, -1))[0]
    top3_indices = np.argsort(prediction)[-3:][::-1]
    top3_probs = prediction[top3_indices]
    top3_labels = [class_dict[idx + 1] for idx in top3_indices]

    modified_image = draw_landmarks_on_image(image, detection_result)
    return modified_image, top3_labels, top3_probs


# Set up the Streamlit interface
st.title('Emotion Recognition Model')
st.write("Upload an image or use your webcam to capture live feed.")

# Option to choose between File Uploader or Webcam
option = st.selectbox('Select input source:', ['Upload Image', 'Use Webcam'])


if option == 'Upload Image':
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        if st.button('Predict'):
            modified_image, labels, probabilities = predict(image, model)
            st.image(modified_image, caption='Modified Image', use_column_width=True)
            st.write({label: f"{prob:.2f}" for label, prob in zip(labels, probabilities)})
else:
    image_data = st.camera_input("Take a picture")
    if image_data is not None:
        image = Image.open(image_data)
        image = np.array(image)
        if st.button('Predict'):
            modified_image, labels, probabilities = predict(image)
            st.image(modified_image, caption='Captured Image', use_column_width=True)
            st.write({label: f"{prob:.2f}" for label, prob in zip(labels, probabilities)})
