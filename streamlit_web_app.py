import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"   #run on CPU to ease deployment


def happinessMean(aus):
    happiness_aus = [6, 12]
    return round(sum([aus.count(x) for x in happiness_aus])/len(happiness_aus), 2)


def sadnessMean(aus):
    sadness_aus = [1, 4, 15]
    return round(sum([aus.count(x) for x in sadness_aus])/len(sadness_aus), 2)


def surpriseMean(aus):
    surprise_aus = [1, 2, 5, 26]
    return round(sum([aus.count(x) for x in surprise_aus])/len(surprise_aus), 2)


def fearMean(aus):
    fear_aus = [1, 2, 4, 5, 7, 20, 26]
    return round(sum([aus.count(x) for x in fear_aus])/len(fear_aus), 2)


def angerMean(aus):
    anger_aus = [4, 5, 7, 23]
    return round(sum([aus.count(x) for x in anger_aus])/len(anger_aus), 2)


def disgustMean(aus):
    disgust_aus = [9, 15, 16]
    return round(sum([aus.count(x) for x in disgust_aus])/len(disgust_aus), 2)


def contemptMean(aus):
    contempt_aus = [12, 14]
    return round(sum([aus.count(x) for x in contempt_aus])/len(contempt_aus), 2)


def get_img_emotions_occurencies(img_aus):
    emotions = dict()
    img_aus = list(img_aus)
    emotions["happiness"] = happinessMean(img_aus)
    emotions["sadness"] = sadnessMean(img_aus)
    emotions["surprise"] = surpriseMean(img_aus)
    emotions["fear"] = fearMean(img_aus)
    emotions["anger"] = angerMean(img_aus)
    emotions["disgust"] = disgustMean(img_aus)
    emotions["contempt"] = contemptMean(img_aus)
    return emotions


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

    simulate = False

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

            # qui dare faces[0] in input alla CNN per riconoscere le AUs
            # qui sotto per ora simulo tutto
            if len(faces) > 0:

                if simulate:
                    img_aus = np.random.choice(aus, 5)
                    emotions_dict = get_img_emotions_occurencies(img_aus)
                    t.text(f"Detected action units: {img_aus}\n\nDetected emotion: {max(emotions_dict, key=emotions_dict.get)}")

                else:
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