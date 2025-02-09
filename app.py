import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Define the model predictions functions
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("trained_potato_leaf_disease_model.keras")

def model_prediction(test_image):
    """Function to load the trained model and predict the class of the uploaded image."""
    model = load_model()
    
    # Convert the PIL image to a file-like object
    img_byte_arr = io.BytesIO()
    test_image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    
    image = tf.keras.preprocessing.image.load_img(img_byte_arr, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# Sidebar Navigation
st.sidebar.title("Home Page...")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Detection"])

# Home Page
if app_mode == "Home":
    st.header("Potato Leaf Disease Detection System")
    st.image("Potato Image.jpeg", use_column_width=True)
    st.markdown("""
    ## Welcome to the Potato Leaf Disease Detection System! 🥔🌿
    
    This AI-based system helps farmers and researchers detect **potato leaf diseases** accurately and efficiently. Just **upload an image** of a potato leaf, and our model will analyze it to determine whether the leaf is **healthy** or affected by **Early Blight** or **Late Blight**.
    
    ### How It Works:
    1. **Upload an Image:** Navigate to the **Disease Detection** page and upload a potato leaf image.
    2. **Processing:** The model analyzes the image using deep learning techniques.
    3. **Result:** The system predicts the disease (if any) and displays the result.
    
    ### Why This Project?
    - **Early Detection:** Helps prevent crop loss by identifying diseases at an early stage.
    - **AI-Based Approach:** Uses deep learning for fast and accurate results.
    - **User-Friendly:** Simple interface for easy access.
    
    ➡️ Go to **Disease Detection** in the sidebar to try it now!
    """)

# About Page
elif app_mode == "About":
    st.header("About the Project")
    st.markdown("""
    ### Overview
    Potato plants are highly vulnerable to diseases, especially **Early Blight** and **Late Blight**, which can cause severe damage to crops. This project aims to build a **deep learning-based disease detection system** to help farmers quickly and accurately identify infected leaves.
    
    ### Dataset Details
    - **Source:** Kaggle's Potato Leaf Disease Dataset
    - **Classes:**
      - **Potato___Healthy**
      - **Potato___Early_Blight**
      - **Potato___Late_Blight**
      
    ### Technology Used
    - **TensorFlow & Keras:** Model training
    - **OpenCV & PIL:** Image preprocessing
    - **Streamlit:** Web app development
    - **NumPy & Matplotlib:** Data analysis & visualization
    
    This project provides a practical solution to detect and prevent the spread of **potato leaf diseases** in agriculture.
    """)

# Disease Detection Page
if app_mode == "Disease Detection":
    st.title("Potato Leaf Disease Detection")
    
    uploaded_file = st.file_uploader("Upload a potato leaf image to detect the disease...", type=["jpg", "png", "jpeg"])

    # Store the uploaded image in session state
    if uploaded_file is not None:
        st.session_state.image = Image.open(uploaded_file)

    # Show Image Button
    if st.button("Show Image"):
        if "image" in st.session_state:
            st.image(st.session_state.image, caption="Uploaded Image", use_column_width=True)

    # Predict Button
    if st.button("Predict"):
        if "image" in st.session_state:
            st.snow()
            st.write("Analyzing... 🔍")
            
            result_index = model_prediction(st.session_state.image)
            
            # Defining class labels
            class_names = ['Potato___Early_Blight', 'Potato___Late_Blight', 'Potato___Healthy']
            prediction = class_names[result_index]
            
            # Display both the image and prediction
            st.image(st.session_state.image, caption=f"Prediction: {prediction}", use_column_width=True)
            st.success(f"Prediction: {prediction}")
