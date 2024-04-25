# importing the libraries
import os
import gradio as gr
import cv2
from PIL import Image
import glob
import random
import numpy as np
import pandas as pd

import mediapipe as mp

from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')


#### Data Preparation
# getting the landmarks given an image path
def get_landmarks(image):
    model_path = 'models/face_landmarker.task'
    
    # STEP 1: Creating an FaceLandmarker object.
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE)
    
    # STEP 2: Loading the input image.
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    
    # STEP 3: Detecting face landmarks from the input image.
    with FaceLandmarker.create_from_options(options) as landmarker:
        detection_result = landmarker.detect(image)

    return detection_result


# process data
def get_process_landmarks(detection_results):
    landmarks = []
    if len(detection_results.face_landmarks)>0:
        for landmark in detection_results.face_landmarks[0]:
            x, y = landmark.x, landmark.y
            landmarks.append([x, y])
        return np.array(landmarks)
    else:
        return None
    

def get_features(landmarks):
    
    # Normalize features by the inter-ocular distance (distance between eyes)
    eye_left = np.array(landmarks[33])  # left eye corner
    eye_right = np.array(landmarks[263]) # right eye corner
    inter_ocular_distance = np.linalg.norm(eye_left - eye_right)
    
    features = []
    
    # Eye features: eye size could be approximated by width and height
    eye_left_top = np.array(landmarks[159])
    eye_left_bottom = np.array(landmarks[145])
    eye_right_top = np.array(landmarks[386])
    eye_right_bottom = np.array(landmarks[374])
    left_eye_height = np.linalg.norm(eye_left_top - eye_left_bottom)
    right_eye_height = np.linalg.norm(eye_right_top - eye_right_bottom)
    left_eye_width = np.linalg.norm(eye_left - np.array(landmarks[133]))
    right_eye_width = np.linalg.norm(eye_right - np.array(landmarks[362]))
    
    # Normalize and append eye features
    features.append(left_eye_height / inter_ocular_distance)
    features.append(right_eye_height / inter_ocular_distance)
    features.append(left_eye_width / inter_ocular_distance)
    features.append(right_eye_width / inter_ocular_distance)
    
    # Lip size: using width and height of the lips
    lips_top = np.array(landmarks[13])
    lips_bottom = np.array(landmarks[14])
    lips_left = np.array(landmarks[61])
    lips_right = np.array(landmarks[291])
    lips_height = np.linalg.norm(lips_top - lips_bottom)
    lips_width = np.linalg.norm(lips_left - lips_right)
    
    # Normalize and append lip features
    features.append(lips_height / inter_ocular_distance)
    features.append(lips_width / inter_ocular_distance)
    
    # Nose size: length and width of the nose
    nose_tip = np.array(landmarks[4])
    nose_top = np.array(landmarks[6])
    nose_left = np.array(landmarks[107])
    nose_right = np.array(landmarks[336])
    nose_height = np.linalg.norm(nose_tip - nose_top)
    nose_width = np.linalg.norm(nose_left - nose_right)
    
    # Normalize and append nose features
    features.append(nose_height / inter_ocular_distance)
    features.append(nose_width / inter_ocular_distance)
    
    return np.array(features)


# loading the data
def load_data(directory):
    labels = []
    features = []

    image_paths = (glob.glob(os.path.join(directory, "**/*")))
    random.shuffle(image_paths)
    image_paths = image_paths[:100]
    
    for img_path in image_paths:
        image = Image.open(img_path)
        image = np.array(image)
        
        landmarks = get_landmarks(image)
        feature = get_process_landmarks(landmarks)
        
        if feature is not None:
            feature = get_features(feature)
            label = int(img_path.split("/")[-2])
            features.append(feature)
            labels.append(label)

    features = np.array(features)
    return features, np.array(labels)

# Load data
def train_model():
    train_features, train_labels = load_data('data/DATASET/train')
    test_features, test_labels = load_data('data/DATASET/test')

    # saving the data
    np.save("train_features.npy", train_features)
    np.save("train_lables.npy", train_labels)

    #### Machine Learning Model
    # creating random forest classifier
    forest = RandomForestClassifier(n_estimators=200, min_samples_leaf=50)
    forest.fit(train_features, train_labels)

    print("The train accuracy is :", forest.score(train_features, train_labels))
    print("The test accuracy is :", forest.score(test_features, test_labels))

    # saving the model
    import joblib
    joblib.dump(forest, 'emotion_recognition_model.pkl')


if __name__ == "__main__":
    model = train_model()
    # Your code to save the model or report metrics
    print("Model trained successfully.")
