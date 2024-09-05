import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

# Load the trained model
model_path = 'Saved//fake_real_cnn.h5'
model = load_model(model_path)

# Define the label map
label_map = {0: 'Fake', 1: 'Real'}

# Streamlit app
st.title('Fake vs Real Image Classification')
st.write('Upload an image to classify if it is fake or real.')

# File uploader
uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Open the image file
    image = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write('')
    st.write('Classifying...')

    # Preprocess the image
    img_array = np.array(image)
    # img_array = cv2.resize(img_array, (256, 256))
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Display the result
    st.write(f'This image is: **{label_map[predicted_class]}**')
