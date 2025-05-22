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


# --- Load the Model ---
@st.cache_resource
def load_ml_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading the ML model from '{MODEL_PATH}': {e}")
        st.stop()


model = load_ml_model()

# --- Define Class Names ---
# You need to know the order of classes your model was trained on.
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
def preprocess_image(image):
    # Ensure the image is in RGB format
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize the image to the target size expected by your model
    target_size = (128, 128)  # model's input size
    image = image.resize(target_size)

    # Convert the image to a NumPy array
    img_array = np.array(image)

    # Normalize the pixel values (e.g., to 0-1 range if your model expects it)
    # This step is critical and depends on how your model was trained.
    # Common normalization: img_array = img_array / 255.0
    img_array = img_array / 255.0  # Example normalization

    # Add an extra dimension for the batch (model expects a batch of images)
    # Input shape will be (1, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# --- Streamlit Application Content ---

st.title("AgroDetect: Aplikasi Identifikasi Hama dan Penyakit Daun")
st.markdown(
    "Identifikasi hama dan penyakit pada daun paprika, tomat, dan kentang menggunakan Machine Learning."
)

# --- About Section ---
st.header("Tentang AgroDetect")
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
)

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang Diunggah", use_container_width=True)

    if st.button("Lakukan Identifikasi"):
        with st.spinner("Menganalisis gambar..."):
            try:
                # Preprocess the image
                processed_image = preprocess_image(image)

                # Make prediction
                predictions = model.predict(processed_image)

                # Get the predicted class (index with highest probability)
                predicted_class_index = np.argmax(predictions, axis=1)[0]
                predicted_class_name = CLASS_NAMES[predicted_class_index]
                confidence = predictions[0][predicted_class_index] * 100

                st.subheader("Hasil Identifikasi:")
                st.success(f"**Identifikasi:** {predicted_class_name}")
                st.info(f"**Keyakinan Model:** {confidence:.2f}%")
                st.info(
                    "Klasifikasi ini didasarkan pada analisis fitur visual dari foto yang diunggah pengguna, seperti pola bercak dan perubahan warna, untuk identifikasi dini yang akurat dan cepat."
                )

                st.write("---")
                st.subheader("Detail Prediksi (Probabilitas per Kelas):")
                # Display all class probabilities for more transparency
                for i, prob in enumerate(predictions[0]):
                    st.write(f"- {CLASS_NAMES[i]}: {prob * 100:.2f}%")

            except Exception as e:
                st.error(
                    f"Terjadi kesalahan saat memproses gambar atau memprediksi: {e}"
                )
                st.info(
                    "Pastikan gambar yang diunggah sesuai dan model Anda sudah dilatih dengan benar."
                )

# --- Footer ---
st.markdown("---")
st.markdown("Ditenagai oleh Laskar Ai, Ai Merdeka Lintasarta, NVIDIA, dan Dicoding.")
st.markdown("ID Grup: LAI25-RM097")
st.markdown("Anggota Grup:")
st.markdown("1. A180YBF358 - Mukhamad Ikhsanudin - Universitas Airlangga")
st.markdown(
    "2. A706YBF391 - Patuh Rujhan Al Istizhar - Universitas Swadaya Gunung Jati"
)
st.markdown("3. A573YBF408 - Rahmat Hidayat - Universitas Lancang Kuning")
st.markdown("4. A314YBF428 - Rifzki Adiyaksa - Universitas Singaperbangsa Karawang")
