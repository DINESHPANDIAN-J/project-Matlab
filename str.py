import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the saved model
model = load_model('rice_leaf_3_7_model.h5')

# Create a dictionary to map class indices to labels
class_indices = {0: 'Nitrogen', 1: 'Phosphorus', 2: 'Potassium', 3: 'SR', 4: 'bacterial_leaf_blight', 5: 'brown_spot', 6: 'healthy', 7: 'leaf_scald', 8: 'narrow_brown_spot'}

# Function to preprocess the image
def preprocess_image(image):
    image = load_img(image, target_size=(256, 256))  # Load the image with target size
    image = img_to_array(image)  # Convert the image to array
    image = np.expand_dims(image, axis=0)  # Expand dimensions to match the input shape
    image = image / 255.0  # Rescale the image
    return image

# Function to make a prediction and get the label
def predict_image(image):
    image = preprocess_image(image)
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = class_indices[predicted_class]
    return predicted_label

# Streamlit App
st.title("Rice Leaf Disease Classification")

st.write("Upload an image of a rice leaf and the model will predict its disease category.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = load_img(uploaded_file, target_size=(256, 256))
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Make a prediction
    predicted_label = predict_image(uploaded_file)
    st.write(f"Predicted label: {predicted_label}")
