import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="AgroDetect: Aplikasi Identifikasi Hama dan Penyakit Daun",
    # icon_image="path/to/your/icon.png"
)

# --- Configuration ---
MODEL_PATH = "best_model.keras"
CONFIDENCE_THRESHOLD = 80


# --- Load the Model ---
@st.cache_resource
def load_ml_model():
    """Loads the Keras model from the specified path."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading the ML model from '{MODEL_PATH}': {e}")
        st.warning(
            "Please ensure the model file 'best_model.keras' is in the same directory as the script, or provide the correct path."
        )
        st.stop()


model = load_ml_model()

# --- Define Class Names ---
# This order must match the order of classes the model was trained on.
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


# --- Preprocessing Function ---
def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocesses the uploaded image to be suitable for model prediction.
    Args:
        image: A PIL Image object.
    Returns:
        A NumPy array representing the processed image.
    """
    # Ensure the image is in RGB format
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize the image to the target size expected by the model
    target_size = (128, 128)
    image = image.resize(target_size)

    # Convert the image to a NumPy array
    img_array = np.array(image)

    # Normalize the pixel values to the 0-1 range
    img_array = img_array / 255.0

    # Add an extra dimension for the batch
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# --- Streamlit Application Content ---

st.title("AgroDetect: Aplikasi Identifikasi Hama dan Penyakit Daun")
st.markdown(
    "Identifikasi hama dan penyakit pada daun paprika, tomat, dan kentang menggunakan Machine Learning."
)

# --- About Section ---
with st.expander("Tentang AgroDetect", expanded=False):
    st.write(
        """
        AgroDetect hadir sebagai solusi berupa aplikasi berbasis web yang dirancang untuk
        memfasilitasi petani dalam mengidentifikasi hama dan penyakit pada daun paprika, tomat,
        dan kentang secara otomatis. Memanfaatkan teknologi Machine Learning, aplikasi ini
        bertujuan untuk memberikan identifikasi yang cepat, akurat, dan efisien langsung di lokasi
        pertanian, bahkan dalam kondisi real-time.
        """
    )

# --- Main Application: Image Upload and Prediction ---
st.header("Identifikasi Hama dan Penyakit")

uploaded_file = st.file_uploader(
    "Unggah gambar daun tanaman (paprika, tomat, atau kentang) di sini:",
    type=["jpg", "jpeg", "png"],
    help="Pastikan gambar jelas dan fokus pada daun yang terindikasi.",
)

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang Diunggah", use_container_width=True)

    if st.button("Lakukan Identifikasi", key="identify_button"):
        with st.spinner("Menganalisis gambar... Mohon tunggu sebentar."):
            try:
                # Preprocess the image
                processed_image = preprocess_image(image)

                # Make prediction
                predictions = model.predict(processed_image)

                # Get the predicted class (index with highest probability)
                predicted_class_index = np.argmax(predictions, axis=1)[0]
                confidence = predictions[0][predicted_class_index] * 100

                st.subheader("Hasil Identifikasi:")

                if confidence >= CONFIDENCE_THRESHOLD:
                    predicted_class_name = CLASS_NAMES[predicted_class_index]
                    st.success(f"**Identifikasi:** {predicted_class_name}")
                    st.info(f"**Keyakinan Model:** {confidence:.2f}%")
                    st.markdown(
                        """
                        <div style="background-color: #e6f7ff; border-left: 5px solid #1890ff; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                        Klasifikasi ini didasarkan pada analisis fitur visual dari foto yang diunggah pengguna,
                        seperti pola bercak dan perubahan warna, untuk identifikasi dini yang akurat dan cepat.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.warning(
                        "**Identifikasi:** Tidak dapat mengidentifikasi penyakit/hama dengan pasti."
                    )
                    st.info(
                        f"**Keyakinan Model (Tertinggi):** {confidence:.2f}% (di bawah ambang batas {CONFIDENCE_THRESHOLD}%)"
                    )
                    st.markdown(
                        """
                        <div style="background-color: #fffbe6; border-left: 5px solid #faad14; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                        Model tidak cukup yakin dengan hasil identifikasi. Coba unggah gambar yang lebih jelas atau dari sudut yang berbeda.
                        Pastikan gambar fokus pada area daun yang menunjukkan gejala.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.write("---")
                # Display all class probabilities for more transparency, regardless of threshold
                with st.expander(
                    "Detail Prediksi (Probabilitas per Kelas)", expanded=False
                ):
                    # Sort probabilities for better readability
                    sorted_indices = np.argsort(predictions[0])[
                        ::-1
                    ]  # Sort in descending order
                    for i in sorted_indices:
                        prob = predictions[0][i] * 100
                        class_name = CLASS_NAMES[i]
                        # Highlight the top prediction if it was above threshold
                        if (
                            i == predicted_class_index
                            and confidence >= CONFIDENCE_THRESHOLD
                        ):
                            st.markdown(
                                f"- **{class_name}: {prob:.2f}% (Teridentifikasi)**"
                            )
                        elif (
                            i == predicted_class_index
                            and confidence < CONFIDENCE_THRESHOLD
                        ):
                            st.markdown(
                                f"- *{class_name}: {prob:.2f}% (Keyakinan tertinggi, namun di bawah ambang batas)*"
                            )
                        else:
                            st.write(f"- {class_name}: {prob:.2f}%")

            except AttributeError as ae:
                # This can happen if the model object is None (failed to load)
                st.error(
                    f"Terjadi kesalahan: Model tidak berhasil dimuat. Detail: {ae}"
                )
                st.info("Pastikan file model 'best_model.keras' ada dan tidak rusak.")
            except Exception as e:
                st.error(
                    f"Terjadi kesalahan saat memproses gambar atau memprediksi: {e}"
                )
                st.info(
                    "Pastikan gambar yang diunggah sesuai dan model Anda sudah dilatih dengan benar."
                )

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #888;">
        <p>Ditenagai oleh Laskar Ai, Ai Merdeka Lintasarta, NVIDIA, dan Dicoding.</p>
        <p>ID Grup: LAI25-RM097</p>
        <p>Anggota Grup:</p>
        <ul style="list-style-type: none; padding: 0;">
            <li>A180YBF358 - Mukhamad Ikhsanudin - Universitas Airlangga</li>
            <li>A706YBF391 - Patuh Rujhan Al Istizhar - Universitas Swadaya Gunung Jati</li>
            <li>A573YBF408 - Rahmat Hidayat - Universitas Lancang Kuning</li>
            <li>A314YBF428 - Rifzki Adiyaksa - Universitas Singaperbangsa Karawang</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
<<<<<<< HEAD
)
=======
)
>>>>>>> db4dcfbd12e11b312ddd5e3294663a94333b640e
