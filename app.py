import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image as PILImage # To avoid conflict with tensorflow.keras.preprocessing.image

# --- Instructions --- 
# To run this Streamlit app:
# 1. First, make sure you have Streamlit installed:
#    pip install streamlit
# 2. Save this code into a Python file, e.g., `app.py`.
# 3. Run it from your terminal using:
#    streamlit run app.py
# --- End Instructions ---

# Load the trained model
# Make sure the model file 'cacao_disease_mobilenetv2_finetuned_model.keras' is in the same directory
# or provide the full path to it.
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('cacao_disease_mobilenetv2_finetuned_model.keras')
    return model

model = load_model()

# Define image dimensions and class names (ensure these match your training setup)
IMG_HEIGHT = 224
IMG_WIDTH = 224
class_names = ['Fito', 'Monilia', 'Sana'] # Based on train_generator.class_indices: {'Fito': 0, 'Monilia': 1, 'Sana': 2}

st.title("Cacao Disease Prediction App")
st.write("Upload an image of a cacao leaf to predict its disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = PILImage.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("")

    # Preprocess the image for prediction
    img = img.resize((IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Create a batch
    img_array = img_array / 255.0 # Rescale the image like during training

    # Make prediction
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    st.subheader("Prediction:")
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    st.write(f"The image most likely belongs to **{predicted_class}** with a **{confidence:.2f}%** confidence.")

    st.markdown("--- More Details ---")
    # Display all class probabilities
    for i, class_name in enumerate(class_names):
        st.write(f"{class_name}: {100 * score[i]:.2f}%")

else:
    st.info("Please upload an image to get a prediction.")
