import streamlit as st
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
from PIL import Image
# import cv2
# from size_extract import size_extract


# st.set_page_config(page_title="Mango Maturity Classifier", page_icon=":mango:")


st.sidebar.write("Menu")

st.title("Mango Ripeness Classifier")

st.write("## This app classifies mango fruit into three ripeness stages and estimates the size")
st.write(" #### Ripe, Very ripe and Unripe")


img_height = 180
img_width = 180
class_names = ['ripe', 'unripe', 'very ripe']
model_path = "fruit_model_new.h5"

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
# uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
# st.sidebar.file_uploader("upload file")

st.set_option("deprecation.showfileUploaderEncoding",False)
# Load the model
@st.cache(allow_output_mutation=True)
def load_model():
    return tf.keras.models.load_model(model_path)

model = load_model()

# Load and preprocess the image
def preprocess_image(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array,0)
    return img_array

# Make predictions
def make_predictions(image_path, model):
    img_array = preprocess_image(image_path)
    print("Input image path:", image_path)
    print("Preprocessed image array:", img_array)
    predictions = model.predict(img_array)
    print("Predictions:", predictions)
    score = tf.nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    return predicted_class, confidence


if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Predict button
    if st.button("Predict"):
        # Make prediction
        predicted_class, confidence = make_predictions(uploaded_file, model)
        col1,col2 = st.columns(2)

        with col1:
            st.write(f"Predicted class: **{predicted_class}**")
            st.write(f"Confidence: **{confidence:.2f}%**")
       

        with col2:
             st.write("width:")
             st.write("Length:")

