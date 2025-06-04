import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# --- Page Configuration ---
st.set_page_config(
    page_title="AgroDetect: Smart Garden Assistant",
    page_icon="🌱",
    layout="wide",
)

# --- ML Model Path & Confidence Threshold ---
MODEL_PATH = "best_model.keras"  # Ensure this model file is in the same directory
CONFIDENCE_THRESHOLD = 75  # Confidence threshold for specific disease/health status results
VERIFICATION_THRESHOLD = 80 # [Baru] Confidence threshold for verifying if it's a correct plant image

# --- Disease & Health Information (Translated to English) ---
CLASS_NAMES = [
    "Pepper_bell__Bacterial_spot",
    "Pepper_bell__healthy",
    "Potato_Early_blight",
    "Potato_Late_blight",
    "Potato_healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_mite",
    "Tomato_Target_Spot",
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_Tomato_mosaic_virus",
    "Tomato_healthy",
]

disease_info = {
    "Pepper_bell__Bacterial_spot": {
        "display_name": "Pepper Bell Bacterial Spot",
        "brief_description": "Bacteria cause oily spots and lesions on leaves and fruits.",
        "cause": "Bacteria _Xanthomonas campestris_. Spreads via water, wind, tools.",
        "symptoms": [
            "Small, dark, oily spots with a yellow halo on leaves.",
            "Crusty lesions on fruits.",
        ],
        "solutions": [
            "Use healthy seeds.",
            "Crop rotation.",
            "Garden sanitation.",
            "Avoid overhead watering.",
            "Copper-based bactericides.",
        ],
    },
    "Pepper_bell__healthy": {
        "display_name": "Healthy Pepper Bell",
        "brief_description": "Your pepper plant is in excellent condition.",
        "cause": "Good cultivation practices.",
        "symptoms": ["Bright green, strong leaves.", "No spots or discoloration."],
        "solutions": ["Maintain routine care.", "Continuously monitor plant health."],
    },
    "Potato_Early_blight": {
        "display_name": "Potato Early Blight",
        "brief_description": "Fungus _Alternaria solani_ causes concentric spots on leaves.",
        "cause": "Fungus _Alternaria solani_. Survives on plant debris.",
        "symptoms": [
            "Brown, circular spots with a target-like pattern on older leaves."
        ],
        "solutions": [
            "Resistant varieties.",
            "Crop rotation.",
            "Remove plant debris.",
            "Fungicides.",
        ],
    },
    "Potato_Late_blight": {
        "display_name": "Potato Late Blight",
        "brief_description": "A fast-spreading fungal disease damaging leaves and tubers.",
        "cause": "Fungus _Phytophthora infestans_. Thrives in cool temperatures, high humidity.",
        "symptoms": [
            "Dark, water-soaked spots on leaves & stems.",
            "White mold on the underside of leaves.",
            "Tuber rot.",
        ],
        "solutions": [
            "Healthy seed potatoes.",
            "Resistant varieties.",
            "Good air circulation.",
            "Avoid overhead watering.",
            "Systemic/contact fungicides.",
        ],
    },
    "Potato_healthy": {
        "display_name": "Healthy Potato",
        "brief_description": "Your potato plant is growing well and is disease-free.",
        "cause": "Optimal environment, proper management.",
        "symptoms": [
            "Dark green leaves, strong growth.",
            "No signs of diseases/pests.",
        ],
        "solutions": ["Continue routine care.", "Monitor and maintain field cleanliness."],
    },
    "Tomato_Bacterial_spot": {
        "display_name": "Tomato Bacterial Spot",
        "brief_description": "Bacteria cause spots on tomato leaves, stems, and fruits.",
        "cause": "Bacteria _Xanthomonas_ species. Spreads through water splashes, seeds.",
        "symptoms": [
            "Small, watery, dark spots on leaves.",
            "Raised, scabby spots on fruits.",
        ],
        "solutions": [
            "Disease-free seeds/seedlings.",
            "Sanitation.",
            "Crop rotation.",
            "Avoid wetting leaves.",
            "Copper-based bactericides.",
        ],
    },
    "Tomato_Early_blight": {
        "display_name": "Tomato Early Blight",
        "brief_description": "Fungus _Alternaria solani_ causes spots with concentric rings.",
        "cause": "Fungus _Alternaria solani_. Survives on plant debris.",
        "symptoms": ["Brown, circular spots with concentric rings on older leaves."],
        "solutions": [
            "Resistant varieties.",
            "Crop rotation.",
            "Clean up plant debris.",
            "Fungicides.",
        ],
    },
    "Tomato_Late_blight": {
        "display_name": "Tomato Late Blight",
        "brief_description": "A fast-spreading fungal disease that damages all parts of the plant.",
        "cause": "Fungus _Phytophthora infestans_. Favors cool temperatures, high humidity.",
        "symptoms": [
            "Large, dark, water-soaked spots.",
            "White mold on the underside of leaves.",
            "Fruits rot.",
        ],
        "solutions": [
            "Resistant varieties.",
            "Healthy seedlings.",
            "Ensure good air circulation.",
            "Avoid overhead watering.",
            "Fungicides.",
        ],
    },
    "Tomato_Leaf_Mold": {
        "display_name": "Tomato Leaf Mold",
        "brief_description": "Fungus causes a velvety layer under leaves, especially in humid environments.",
        "cause": "Fungus _Passalora fulva_ (syn. _Fulvia fulva_). High humidity (>85%), moderate temperatures.",
        "symptoms": [
            "Yellowish-green spots on the upper side of leaves.",
            "Brownish-gray velvety layer on the underside of leaves.",
        ],
        "solutions": [
            "Resistant varieties.",
            "Improve air circulation.",
            "Lower humidity.",
            "Avoid wetting leaves.",
            "Fungicides.",
        ],
    },
    "Tomato_Septoria_leaf_spot": {
        "display_name": "Tomato Septoria Leaf Spot",
        "brief_description": "Fungus causes small, circular spots with dark specks in the center.",
        "cause": "Fungus _Septoria lycopersici_. Spreads through water splashes.",
        "symptoms": [
            "Small, circular, brown spots with a gray center and tiny black dots (pycnidia)."
        ],
        "solutions": [
            "Crop rotation.",
            "Garden sanitation.",
            "Use mulch.",
            "Avoid overhead watering.",
            "Fungicides.",
        ],
    },
    "Tomato_Spider_mites_Two_spotted_mite": {
        "display_name": "Tomato Spider Mites (Two-spotted mite)",
        "brief_description": "Small sap-sucking pests causing yellow spots and fine webs.",
        "cause": "Mite _Tetranychus urticae_. Reproduces quickly in hot, dry conditions.",
        "symptoms": [
            "Yellow/bronze stippling on leaves.",
            "Fine webbing between leaves.",
            "Leaves may curl/dry up.",
        ],
        "solutions": [
            "Maintain humidity.",
            "Spray with pressurized water.",
            "Natural enemies (predatory mites).",
            "Insecticidal soap/neem oil.",
            "Acaricides (miticides).",
        ],
    },
    "Tomato_Target_Spot": {
        "display_name": "Tomato Target Spot",
        "brief_description": "Fungus causes concentric 'target-like' spots on leaves.",
        "cause": "Fungus _Corynespora cassiicola_. Spreads via wind/water.",
        "symptoms": ["Dark brown, circular spots with concentric zones like a target."],
        "solutions": [
            "Crop rotation.",
            "Sanitation.",
            "Good drainage.",
            "Improve air circulation.",
            "Fungicides.",
        ],
    },
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus": {
        "display_name": "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "brief_description": "Virus transmitted by whiteflies, causing severe yellowing and curling of leaves.",
        "cause": "TYLCV virus transmitted by whiteflies (_Bemisia tabaci_).",
        "symptoms": [
            "Leaves yellow between veins, curl upwards.",
            "Stunted plants, reduced fruit set.",
        ],
        "solutions": [
            "Resistant varieties.",
            "Control whiteflies.",
            "Silver reflective mulch.",
            "Sanitation.",
            "Remove infected plants.",
        ],
    },
    "Tomato_Tomato_mosaic_virus": {
        "display_name": "Tomato Mosaic Virus (ToMV)",
        "brief_description": "Highly contagious virus causing mosaic patterns on leaves and stunted growth.",
        "cause": "ToMV virus. Easily transmitted mechanically (touch, tools, seeds).",
        "symptoms": [
            "Mosaic pattern (light/dark green) on leaves.",
            "Curled/deformed leaves.",
            "Stunted plants.",
        ],
        "solutions": [
            "Healthy seeds.",
            "Wash hands & sterilize tools.",
            "Avoid tobacco products near plants.",
            "Remove infected plants.",
        ],
    },
    "Tomato_healthy": {
        "display_name": "Healthy Tomato",
        "brief_description": "Your tomato plant is in prime condition and productive.",
        "cause": "Optimal cultivation practices.",
        "symptoms": [
            "Dark green leaves, upright growth.",
            "No signs of diseases/pests.",
        ],
        "solutions": [
            "Maintain care.",
            "Regular inspection.",
            "Optimize air circulation.",
            "Regular pruning.",
        ],
    },
}


# --- Image Preprocessing Function ---
@st.cache_data
def preprocess_image(_image: Image.Image) -> np.ndarray:
    """Processes the uploaded image for model prediction."""
    if _image.mode != "RGB":
        _image = _image.convert("RGB")
    target_size = (128, 128)
    _image = _image.resize(target_size)
    img_array = np.array(_image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# --- Cache ML Model ---
@st.cache_resource
def load_ml_model():
    """Loads the pre-trained TensorFlow/Keras model."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(
            f"❌ **Oops!** The Machine Learning model failed to load from '{MODEL_PATH}'. Error: {e}"
        )
        st.warning(
            "This might happen if the model file is missing or corrupted. "
            f"Ensure `best_model.keras` is in the correct location."
        )
        st.stop()


model = load_ml_model()


# --- Function to Reset App State ---
def reset_app_state():
    """Resets all relevant session states to clear results and allow new uploads."""
    st.session_state.uploaded_file = None
    st.session_state.identification_done = False
    st.session_state.predicted_class_name_state = None
    st.session_state.confidence_state = None
    st.session_state.predictions_state = None
    st.session_state.show_detailed_solution = False
    st.session_state.threshold_message = None # Reset threshold message
    st.session_state.file_uploader_key = str(np.random.rand())


# --- Initialize Session State ---
if "identification_done" not in st.session_state:
    st.session_state.identification_done = False
    st.session_state.predicted_class_name_state = None
    st.session_state.confidence_state = None
    st.session_state.predictions_state = None
    st.session_state.show_detailed_solution = False
    st.session_state.current_page = "Identification"
    st.session_state.file_uploader_key = "initial"
    st.session_state.uploaded_file = None
    st.session_state.threshold_message = None


# --- HEADER AND TOP NAVIGATION (CENTERED) ---
header_container = st.container()
with header_container:
    st.markdown(
        f"""
        <div style="display: flex; flex-direction: column; align-items: center; margin-bottom: 20px;">
            <img src="https://emojigraph.org/media/apple/leafy-green_1f96c.png" alt="AgroDetect Logo" width="80">
            <h1 style="text-align: center; margin-bottom: 0px;">🌱 AgroDetect</h1>
            <p style="text-align: center; margin-top: 0px; font-style: italic;">_Your Smart Garden Assistant_</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    nav_placeholder_left, nav_col1, nav_col2, nav_col3, nav_placeholder_right = st.columns(
        [0.5, 1.5, 1.5, 1.5, 0.5]
    )

    with nav_col1:
        if st.button(
            "🏡 Plant Identification",
            key="nav_identification",
            use_container_width=True,
            type="primary" if st.session_state.current_page == "Identification" else "secondary",
        ):
            st.session_state.current_page = "Identification"
            reset_app_state()
            st.rerun()

    with nav_col2:
        if st.button(
            "💡 About AgroDetect",
            key="nav_about",
            use_container_width=True,
            type="primary" if st.session_state.current_page == "About" else "secondary",
        ):
            st.session_state.current_page = "About"
            st.rerun()

    with nav_col3:
        if st.button(
            "👥 Development Team",
            key="nav_team",
            use_container_width=True,
            type="primary" if st.session_state.current_page == "Team" else "secondary",
        ):
            st.session_state.current_page = "Team"
            st.rerun()
st.divider()


# --- MAIN PAGE CONTENT BASED ON NAVIGATION ---

if st.session_state.current_page == "Identification":
    st.markdown(
        "<h1 style='text-align: center; color: #4CAF50;'>🌱 AgroDetect: Identify Your Plant's Issue!</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; font-size: 1.1em;'>Upload an image of your pepper, tomato, or potato plant leaf. "
        "We'll analyze it and provide a quick diagnosis and treatment recommendations.</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.subheader("📸 Upload Leaf Image")
    st.write("Drag & drop an image here, or click to select a file.")

    current_uploaded_file = st.file_uploader(
        "Select a leaf image (JPG, PNG):",
        type=["jpg", "jpeg", "png"],
        key=st.session_state.file_uploader_key,
        label_visibility="collapsed",
    )

    if current_uploaded_file is not None:
        if st.session_state.uploaded_file != current_uploaded_file:
            st.session_state.uploaded_file = current_uploaded_file
            reset_app_state() # Full reset for new file to clear all previous states
            st.session_state.uploaded_file = current_uploaded_file # Re-assign after reset
    elif st.session_state.uploaded_file is not None and current_uploaded_file is None:
        reset_app_state() # Reset if file is removed


    if st.session_state.uploaded_file is not None:
        image = Image.open(st.session_state.uploaded_file)
        st.image(image, caption="Your Leaf Image", use_container_width=True)
    else:
        st.markdown(
            "<div style='border: 2px dashed #4CAF50; padding: 50px; text-align: center; opacity: 0.7;'>"
            "No image uploaded."
            "</div>",
            unsafe_allow_html=True,
        )
        st.info(
            "Upload a clear photo of the leaf for more accurate identification. Focus on the symptomatic area."
        )

    if st.session_state.uploaded_file is not None:
        if st.button(
            "✨ **Start Smart Analysis!**",
            key="analyze_button",
            help="Click to identify the disease on your leaf photo.",
            use_container_width=True,
            type="primary",
        ):
            with st.spinner("⏳ Analysis in progress..."):
                try:
                    st.session_state.threshold_message = None # Reset threshold message for new analysis
                    st.session_state.identification_done = False # Reset identification status

                    processed_image = preprocess_image(
                        Image.open(st.session_state.uploaded_file)
                    )
                    predictions = model.predict(processed_image)
                    predicted_class_index = np.argmax(predictions, axis=1)[0]
                    confidence = predictions[0][predicted_class_index] * 100

                    st.session_state.predictions_state = predictions
                    st.session_state.confidence_state = confidence
                    st.session_state.predicted_class_name_state = CLASS_NAMES[
                        predicted_class_index
                    ]
                    st.session_state.identification_done = True # Set to true after successful prediction

                    if confidence < VERIFICATION_THRESHOLD: # Using new VERIFICATION_THRESHOLD
                        st.session_state.threshold_message = (
                            f"The model could not confidently verify this as a plant leaf from the supported categories, "
                            f"or the object was not clearly detected (confidence: {confidence:.2f}% < {VERIFICATION_THRESHOLD}%). "
                            "Please try uploading a clearer image of a pepper, tomato, or potato leaf."
                        )
                        # Do not proceed to show disease-specific info if below this general verification threshold
                        st.session_state.show_detailed_solution = False # Ensure solution details are not shown

                except Exception as e:
                    st.error(
                        f"❌ **An error occurred during analysis:** {e}. "
                        "Please try again or upload a different image."
                    )
                    st.session_state.identification_done = False
                    st.session_state.threshold_message = None

        # --- DISPLAY IDENTIFICATION RESULTS ---
        if st.session_state.identification_done:
            st.markdown("---")
            st.subheader("💡 Identification Results")

            if st.session_state.threshold_message:
                st.warning(st.session_state.threshold_message)
            else:
                confidence = st.session_state.confidence_state
                predicted_class_name = st.session_state.predicted_class_name_state

                info = disease_info.get(predicted_class_name, {})
                display_name = info.get(
                    "display_name",
                    predicted_class_name.replace("_", " ").replace("__", ": "),
                )
                brief_description = info.get(
                    "brief_description", "Additional information is not available."
                )

                if (
                    confidence >= CONFIDENCE_THRESHOLD # Original threshold for disease/healthy specific confidence
                    and "healthy" not in predicted_class_name.lower()
                ):
                    st.error(f"🚨 Detected: {display_name}")
                    st.metric(
                        label="Confidence Level (Specific Condition)",
                        value=f"{confidence:.2f}%",
                        delta="Disease Detected",
                        delta_color="inverse",
                    )
                    st.markdown(f"**Summary:** {brief_description}")
                    if st.button(
                        "📖 View Detailed Solution & Management",
                        key="view_solution_button",
                        use_container_width=True,
                    ):
                        st.session_state.show_detailed_solution = True

                elif confidence >= CONFIDENCE_THRESHOLD and "healthy" in predicted_class_name.lower():
                    st.success(f"✅ Healthy Plant: {display_name}")
                    st.metric(
                        label="Confidence Level (Specific Condition)",
                        value=f"{confidence:.2f}%",
                        delta="Healthy",
                        delta_color="normal",
                    )
                    st.markdown(f"**Summary:** {brief_description}")
                    if st.button(
                        "💚 Tips for Maintaining Plant Health",
                        key="view_healthy_tips_button",
                        use_container_width=True,
                    ):
                        st.session_state.show_detailed_solution = True
                else: # Confidence for specific condition is below CONFIDENCE_THRESHOLD (75%)
                    st.warning(f"❓ Low Confidence for: {display_name}")
                    st.metric(
                        label="Highest Confidence (Specific Condition)",
                        value=f"{confidence:.2f}%",
                        delta=f"Below {CONFIDENCE_THRESHOLD}% for this specific condition",
                        delta_color="off",
                    )
                    st.write(
                        "The model identified a potential condition but with lower confidence. "
                        "For more certainty, please ensure the image is clear or consult an expert."
                    )
                    # Optionally, still allow viewing details for low confidence specific predictions
                    if st.button(
                        f"📖 View Potential Details for {display_name}",
                        key="view_low_confidence_solution_button",
                        use_container_width=True,
                    ):
                        st.session_state.show_detailed_solution = True


            st.markdown("---")
            if st.button(
                "🔄 **Upload New Image**",
                help="Click to clear the current result and upload another leaf photo.",
                use_container_width=True,
                type="secondary",
            ):
                reset_app_state()
                st.rerun()

    # --- DETAILED SOLUTION SECTION (CONDITIONALLY DISPLAYED) ---
    if (
        st.session_state.get("show_detailed_solution", False)
        and st.session_state.identification_done
        and not st.session_state.threshold_message # Only show if not overridden by general verification threshold
        and st.session_state.predicted_class_name_state # Ensure there's a predicted class
    ):
        st.markdown("---")
        current_display_name = disease_info.get(st.session_state.predicted_class_name_state, {}).get(
            "display_name",
            st.session_state.predicted_class_name_state.replace("_", " ").replace("__", ": ")
        )
        st.header(f"🌿 Detailed Management for {current_display_name}")

        info_detail = disease_info.get(st.session_state.predicted_class_name_state, {})

        if info_detail:
            col_detail_1, col_detail_2 = st.columns(2)
            with col_detail_1:
                with st.expander("📚 **Cause & Typical Symptoms**", expanded=True):
                    st.markdown(
                        f"**Main Cause:** {info_detail.get('cause', 'Not available.')}"
                    )
                    st.markdown("**Symptoms to Look For:**")
                    if isinstance(info_detail.get("symptoms"), list):
                        for symptom in info_detail["symptoms"]:
                            st.markdown(f"- {symptom}")
                    else:
                        st.write(info_detail.get("symptoms", "Not available."))
            with col_detail_2:
                with st.expander("👨‍🌾 **Solution & Management Steps**", expanded=True):
                    st.markdown("**Recommendations:**")
                    if isinstance(info_detail.get("solutions"), list):
                        for solution_step in info_detail["solutions"]:
                            st.markdown(f"- {solution_step}")
                    else:
                        st.write(info_detail.get("solutions", "Not available."))
            st.divider()
            # Only show probabilities if identification was generally successful (not threshold_message)
            if st.session_state.predictions_state is not None and not st.session_state.threshold_message:
                with st.expander("🔬 **Full Probabilities (For Experts)**"):
                    st.write(
                        "Below is the list of model probabilities for each category, from highest to lowest:"
                    )
                    sorted_indices = np.argsort(st.session_state.predictions_state[0])[::-1]
                    for i in sorted_indices:
                        prob = st.session_state.predictions_state[0][i] * 100
                        class_disp = disease_info.get(CLASS_NAMES[i], {}).get(
                            "display_name",
                            CLASS_NAMES[i].replace("_", " ").replace("__", ": "),
                        )
                        if i == np.argmax(st.session_state.predictions_state, axis=1)[0]:
                            st.markdown(f"- **{class_disp}: {prob:.2f}%** (Main Prediction)")
                        else:
                            st.write(f"- {class_disp}: {prob:.2f}%")
        else:
            st.warning("Sorry, detailed information for this result is not available in our database.")


elif st.session_state.current_page == "About":
    st.title("💡 About AgroDetect")
    st.write(
        """
        **AgroDetect** is an innovative web application that empowers modern farmers with the power of **Machine Learning**.
        Our mission is to provide early detection capabilities for pests and diseases on **pepper, tomato, and potato** leaves
        simply through an image upload.
        """
    )
    st.divider()
    st.subheader("Our Vision & Mission")
    st.write(
        """
        **Vision:** To be a leading platform supporting sustainable agriculture through smart AI solutions.
        **Mission:** To provide an accurate and accessible plant disease identification tool, along with practical management recommendations to enhance agricultural productivity.
        """
    )
    st.subheader("Technology Behind the Scenes")
    st.write(
        """
        AgroDetect is built upon an advanced **Convolutional Neural Network (CNN)** model, trained with the extensive and diverse
        **[Plant Village](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)** dataset.
        This enables our model to recognize specific patterns and symptoms of various plant conditions.
        """
    )
    st.info(
        "This application is an initial diagnostic aid and does not replace consultation with professional agricultural experts."
    )

elif st.session_state.current_page == "Team":
    st.title("👨‍💻 Development Team")
    st.write("AgroDetect is the result of a Capstone project by **Team Laskar AI**.")
    st.divider()
    st.subheader("Project Information")
    st.markdown(
        """
        -   **Group ID:** LAI25-RM097
        -   **Theme:** Smart Solutions for a Better Future
        -   **Mentor:** Stevani Dwi Utomo (Mentoring session: June 5, 2025)
        """
    )
    st.subheader("Team Members")
    st.markdown(
        """
        We are individuals passionate about applying AI for real-world solutions:
        -   **Mukhamad Ikhsanudin** (A180YBF358) – Universitas Airlangga
        -   **Patuh Rujhan Al Istizhar** (A706YBF391) – Universitas Swadaya Gunung Jati
        -   **Rahmat Hidayat** (A573YBF408) – Universitas Lancang Kuning
        -   **Rifzki Adiyaksa** (A314YBF428) – Universitas Singaperbangsa Karawang
        """
    )
    st.info("Together, we create innovation for better agriculture.")

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: grey;'>© 2025 AgroDetect. All rights reserved.</p>",
    unsafe_allow_html=True,
)
