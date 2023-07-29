import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"   #run on CPU to ease deployment


@st.cache_resource
def load_ai_model(path):
    return load_model(path)


st.sidebar.title('Emotion recognition from webcam')
start = st.sidebar.checkbox('Open camera', value=True)

t = st.empty()
camera = cv2.VideoCapture(0)




if start:
    t.text("Loading the AI model...")

    frame_window = st.image([])

    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    single_output_model = load_ai_model("models/single_output_model")

    t.text("Opening camera...")

    classes = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
    initialize_counter = 0


    while start:
        success, original_frame = camera.read()

        if success:
            frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB) 
            grayscale_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
            

            faces = face_classifier.detectMultiScale(grayscale_frame, 1.1, 5, minSize=(200, 200))
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

            if initialize_counter > 1:
                frame_window.image(frame)
            else:
                initialize_counter += 1

            if len(faces) > 0:
                # 1) get the colored ROI
                top, right, bottom, left = faces[0]
                roi = frame[right:right + left, top:top + bottom]
                roi = cv2.resize(roi, (200,200), interpolation = cv2.INTER_AREA)

                # 2) transform it to grayscale
                roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

                # 3) repeat on three channels
                roi = roi.reshape((roi.shape[0], roi.shape[1], 1))
                roi = np.repeat(roi, 3, axis=2)

                predictions = single_output_model.predict(np.expand_dims(roi, axis=0))
                top_class = classes[predictions[0].argmax()]

                if initialize_counter > 1:
                    t.text(f"Detected emotion: {top_class}")
            else:
                t.text("I cannot see your face!")

        else:
            t.text("There was a problem with your camera :(")





else:
    t.text("Open the camera to start!")
    camera.release()