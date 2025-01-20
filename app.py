import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import folium
from streamlit_folium import folium_static
from fpdf import FPDF
import time

# Function to load the model and make predictions
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Disease Labels and Descriptions
class_name = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

disease_info = {
    "Apple___Apple_scab": """
        **Apple Scab**  
        - **Cause**: Fungal pathogen *Venturia inaequalis*.  
        - **Symptoms**:  
            - Olive-green or brown velvety spots on leaves.  
            - Dark scabby lesions on fruits leading to fruit drop.  
        - **Prevention**:  
            - Use resistant apple varieties.  
            - Apply fungicides early in the season.
    """,
    "Apple___Black_rot": """
        **Black Rot (Apple)**  
        - **Cause**: Fungus *Botryosphaeria obtusa*.  
        - **Symptoms**:  
            - Dark sunken lesions on fruits.  
            - Leaf browning and branch dieback.  
        - **Prevention**:  
            - Prune infected branches.  
            - Apply copper-based fungicides.
    """,
    "Tomato___Early_blight": """
        **Early Blight (Tomato)**  
        - **Cause**: Fungus *Alternaria solani*.  
        - **Symptoms**:  
            - Dark concentric spots on older leaves.  
            - Premature leaf drop.  
        - **Prevention**:  
            - Crop rotation and fungicide sprays.
    """,
    "Tomato___Late_blight": """
        **Late Blight (Tomato)**  
        - **Cause**: Oomycete *Phytophthora infestans*.  
        - **Symptoms**:  
            - Water-soaked lesions on leaves.  
            - White mold growth under humid conditions.  
        - **Prevention**:  
            - Remove infected plants immediately.  
            - Avoid overhead irrigation.
    """,
    "Tomato___healthy": "The plant appears healthy. No disease detected.",
}

# Sidebar
st.sidebar.title("🌿 Plant Disease Detection")
st.sidebar.markdown("## Navigation")
app_mode = st.sidebar.selectbox("Select Page", ["🏠 HOME", "🔍 DISEASE RECOGNITION"])

# Main Page Layout
if app_mode == "🏠 HOME":
    st.markdown("<h1 style='text-align: center; color: green;'>Plant Disease Detection System</h1>", unsafe_allow_html=True)
    st.image("Diseases.png", use_container_width=True)
    st.markdown("""
        ### 🌾 About the System  
        This system helps farmers detect plant diseases early for timely intervention.  
        Upload a plant leaf image to diagnose its health or detect any diseases.
    """)

elif app_mode == "🔍 DISEASE RECOGNITION":
    st.header("📷 Upload a Leaf Image for Detection")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])

    if test_image and st.button("Show Image"):
        # Add container for the image with optional width
        image_container = st.container()
        with image_container:
            st.image(test_image, caption="Uploaded Image", use_container_width=True)

    # Predict button
    if test_image and st.button("Predict"):
        with st.spinner("Analyzing the image..."):
            time.sleep(2)
            result_index = model_prediction(test_image)
            predicted_disease = class_name[result_index]
            st.success(f"Model Prediction: **{predicted_disease}**")

            # Display disease details
            info = disease_info.get(predicted_disease, "No detailed information available.")
            st.markdown(info)

            # Show progress bar
            progress = st.progress(0)
            for percent in range(100):
                time.sleep(0.01)
                progress.progress(percent + 1)
            st.balloons()

            # Inject Snow Effect using HTML/JS
            snow_js = """
                <script src="https://cdn.jsdelivr.net/npm/snowstorm@1.53.0/snowstorm.js"></script>
                <script type="text/javascript">
                    snowStorm.start();
                </script>
            """
            st.components.v1.html(snow_js, height=0)
