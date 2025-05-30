import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="AgroDetect: Aplikasi Identifikasi Hama dan Penyakit Daun",
    page_icon="🌿",  # Menambahkan ikon halaman
    layout="wide" # Menggunakan layout yang lebih lebar
)

# --- Konfigurasi CSS Kustom (Opsional, untuk styling lebih lanjut) ---
st.markdown("""
<style>
    .stApp {
        # background-color: #f0f2f6; /* Contoh warna latar belakang */
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
    }
    .stButton>button:hover {
        background-color: #45a049;
        color: white;
    }
    .title-app {
        color: #2E8B57; /* Warna hijau tua untuk judul */
        text-align: center;
        font-weight: bold;
    }
    .subtitle-app {
        text-align: center;
        color: #555;
        margin-bottom: 2rem;
    }
    .expander-custom {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)


# --- Configuration ---
MODEL_PATH = "best_model.keras" #
CONFIDENCE_THRESHOLD = 80 #


# --- Load the Model ---
@st.cache_resource
def load_ml_model():
    """Memuat model Keras dari path yang ditentukan."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH) #
        return model
    except Exception as e:
        st.error(f"Error loading the ML model from '{MODEL_PATH}': {e}") #
        st.warning(
            "Pastikan file model 'best_model.keras' berada di direktori yang sama dengan skrip, atau berikan path yang benar." #
        )
        st.stop()


model = load_ml_model() #

# --- Define Class Names ---
# Urutan ini harus sesuai dengan urutan kelas saat model dilatih.
CLASS_NAMES = [ #
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
    Memproses gambar yang diunggah agar sesuai untuk prediksi model.
    Args:
        image: Objek PIL Image.
    Returns:
        Array NumPy yang merepresentasikan gambar yang telah diproses.
    """
    # Pastikan gambar dalam format RGB
    if image.mode != "RGB": #
        image = image.convert("RGB") #

    # Ubah ukuran gambar ke ukuran target yang diharapkan model
    target_size = (128, 128) #
    image = image.resize(target_size) #

    # Konversi gambar ke array NumPy
    img_array = np.array(image) #

    # Normalisasi nilai piksel ke rentang 0-1
    img_array = img_array / 255.0 #

    # Tambahkan dimensi ekstra untuk batch
    img_array = np.expand_dims(img_array, axis=0) #
    return img_array


# --- Sidebar ---
with st.sidebar:
    st.image("https://emojigraph.org/media/apple/leafy-green_1f96c.png", width=100) # Contoh logo/ikon
    st.markdown("<h2 class='title-app' style='text-align: left; font-size: 1.8em;'>AgroDetect</h2>", unsafe_allow_html=True)
    st.markdown("Identifikasi Cepat, Pertanian Hebat!")

    st.markdown("---")
    st.subheader("Menu Navigasi")
    page_selection = st.radio("Pilih Halaman:", ["Identifikasi Tanaman", "Tentang AgroDetect", "Detail Proyek"])
    st.markdown("---")
    st.info("Aplikasi ini menggunakan model Machine Learning untuk mengidentifikasi penyakit pada daun tanaman.")


# --- Streamlit Application Content ---

if page_selection == "Identifikasi Tanaman":
    st.markdown("<h1 class='title-app'>🌿 AgroDetect: Aplikasi Identifikasi Hama dan Penyakit Daun</h1>", unsafe_allow_html=True) #
    st.markdown(
        "<p class='subtitle-app'>Identifikasi hama dan penyakit pada daun paprika, tomat, dan kentang menggunakan Machine Learning.</p>", #
        unsafe_allow_html=True
    )

    # --- Main Application: Image Upload and Prediction ---
    st.header("🔍 Unggah Gambar untuk Identifikasi") #

    uploaded_file = st.file_uploader( #
        "Unggah gambar daun tanaman (paprika, tomat, atau kentang) di sini:", #
        type=["jpg", "jpeg", "png"], #
        help="Pastikan gambar jelas dan fokus pada daun yang terindikasi.", #
    )

    if uploaded_file is not None: #
        col1, col2 = st.columns(2)
        with col1:
            image = Image.open(uploaded_file) #
            st.image(image, caption="Gambar yang Diunggah", use_container_width=True) #

        if st.button("🚀 Lakukan Identifikasi", key="identify_button", help="Klik untuk memulai proses identifikasi"): #
            with st.spinner("⏳ Menganalisis gambar... Mohon tunggu sebentar."): #
                try:
                    # Preprocess the image
                    processed_image = preprocess_image(image) #

                    # Make prediction
                    predictions = model.predict(processed_image) #

                    # Get the predicted class (index with highest probability)
                    predicted_class_index = np.argmax(predictions, axis=1)[0] #
                    confidence = predictions[0][predicted_class_index] * 100 #

                    with col2:
                        st.subheader("📊 Hasil Identifikasi:") #

                        if confidence >= CONFIDENCE_THRESHOLD: #
                            predicted_class_name = CLASS_NAMES[predicted_class_index] #
                            st.success(f"**Identifikasi:** {predicted_class_name}") #
                            st.info(f"**Keyakinan Model:** {confidence:.2f}%") #
                            st.markdown( #
                                """
                                <div style="background-color: #e6f7ff; border-left: 5px solid #1890ff; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                                Klasifikasi ini didasarkan pada analisis fitur visual dari foto yang diunggah pengguna,
                                seperti pola bercak dan perubahan warna, untuk identifikasi dini yang akurat dan cepat.
                                </div>
                                """, #
                                unsafe_allow_html=True,
                            )
                        else:
                            st.warning( #
                                "**Identifikasi:** Tidak dapat mengidentifikasi penyakit/hama dengan pasti." #
                            )
                            st.info( #
                                f"**Keyakinan Model (Tertinggi):** {confidence:.2f}% (di bawah ambang batas {CONFIDENCE_THRESHOLD}%)" #
                            )
                            st.markdown( #
                                """
                                <div style="background-color: #fffbe6; border-left: 5px solid #faad14; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                                Model tidak cukup yakin dengan hasil identifikasi. Coba unggah gambar yang lebih jelas atau dari sudut yang berbeda.
                                Pastikan gambar fokus pada area daun yang menunjukkan gejala.
                                </div>
                                """, #
                                unsafe_allow_html=True,
                            )

                        st.write("---") #
                        # Display all class probabilities for more transparency, regardless of threshold
                        with st.expander( #
                            "Lihat Detail Prediksi (Probabilitas per Kelas)", expanded=False #
                        ):
                            # Sort probabilities for better readability
                            sorted_indices = np.argsort(predictions[0])[ #
                                ::-1
                            ]
                            for i in sorted_indices: #
                                prob = predictions[0][i] * 100 #
                                class_name = CLASS_NAMES[i] #
                                # Highlight the top prediction
                                if ( #
                                    i == predicted_class_index
                                    and confidence >= CONFIDENCE_THRESHOLD #
                                ):
                                    st.markdown( #
                                        f"- **{class_name}: {prob:.2f}% (Teridentifikasi)**" #
                                    )
                                elif ( #
                                    i == predicted_class_index
                                    and confidence < CONFIDENCE_THRESHOLD #
                                ):
                                    st.markdown( #
                                        f"- *{class_name}: {prob:.2f}% (Keyakinan tertinggi, namun di bawah ambang batas)*" #
                                    )
                                else:
                                    st.write(f"- {class_name}: {prob:.2f}%") #

                except AttributeError as ae: #
                    st.error( #
                        f"Terjadi kesalahan: Model tidak berhasil dimuat. Detail: {ae}" #
                    )
                    st.info("Pastikan file model 'best_model.keras' ada dan tidak rusak.") #
                except Exception as e: #
                    st.error( #
                        f"Terjadi kesalahan saat memproses gambar atau memprediksi: {e}" #
                    )
                    st.info( #
                        "Pastikan gambar yang diunggah sesuai dan model Anda sudah dilatih dengan benar." #
                    )
        elif col2.button("Bersihkan", help="Klik untuk menghapus gambar dan hasil"):
            # Ini akan mereset uploaded_file secara implisit pada rerun berikutnya
            # karena tidak ada file yang diunggah
            st.rerun()


elif page_selection == "Tentang AgroDetect":
    st.markdown("<h1 class='title-app'>💡 Tentang AgroDetect</h1>", unsafe_allow_html=True)
    st.markdown( #
        """
        <div class="expander-custom">
        AgroDetect hadir sebagai solusi berupa aplikasi berbasis web yang dirancang untuk
        memfasilitasi petani dalam mengidentifikasi hama dan penyakit pada daun paprika, tomat,
        dan kentang secara otomatis. Memanfaatkan teknologi Machine Learning, aplikasi ini
        bertujuan untuk memberikan identifikasi yang cepat, akurat, dan efisien langsung di lokasi
        pertanian, bahkan dalam kondisi real-time.
        <br><br>
        <strong>Fitur Utama:</strong>
        <ul>
            <li>Identifikasi otomatis penyakit tanaman dari gambar daun.</li>
            <li>Dukungan untuk tanaman paprika, tomat, dan kentang.</li>
            <li>Menampilkan tingkat keyakinan model terhadap hasil identifikasi.</li>
            <li>Antarmuka pengguna yang sederhana dan mudah digunakan.</li>
        </ul>
        <strong>Dataset yang Digunakan:</strong>
        Dataset yang digunakan untuk melatih model adalah <a href="https://www.kaggle.com/datasets/arjuntejaswi/plant-village" target="_blank">Plant Village</a>.
        </div>
        """, #
        unsafe_allow_html=True,
    )

elif page_selection == "Detail Proyek":
    st.markdown("<h1 class='title-app'>👥 Detail Proyek Laskar AI</h1>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="expander-custom">
        Berikut adalah detail mengenai proyek Capstone Laskar AI:
        <br><br>
        <strong>ID Grup:</strong> LAI25-RM097
        <br>
        <strong>Tema yang Dipilih:</strong> Solusi Cerdas untuk Masa Depan yang Lebih Baik
        <br>
        <strong>Nama Advisor:</strong> Stevani Dwi Utomo, [Sesi mentoring dilakukan pada (<em>harap ubah ini dengan tanggal mentoring Anda</em>)]
        <br><br>
        <strong>Anggota Grup:</strong>
        <ul>
            <li>A180YBF358 – Mukhamad Ikhsanudin – Universitas Airlangga - [Aktif]</li>
            <li>A706YBF391 – Patuh Rujhan Al Istizhar – Universitas Swadaya Gunung Jati - [Aktif]</li>
            <li>A573YBF408 – Rahmat Hidayat – Universitas Lancang Kuning - [Aktif]</li>
            <li>A314YBF428 – Rifzki Adiyaksa – Universitas Singaperbangsa Karawang - [Aktif]</li>
        </ul>
        <br>
        <strong>Deskripsi Singkat Proyek:</strong>
        Proyek Laskar AI bertujuan untuk mengembangkan AgroDetect, sebuah aplikasi web
        yang memanfaatkan machine learning untuk membantu petani dalam mengidentifikasi
        hama dan penyakit pada tanaman paprika, tomat, dan kentang. Aplikasi ini
        diharapkan dapat memberikan solusi yang cepat dan akurat sehingga dapat
        meningkatkan efisiensi dan produktivitas pertanian.
        </div>
        """,
        unsafe_allow_html=True
    )


# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #888;">
        © 2024 Tim Laskar AI (LAI25-RM097) | Bangkit Academy 2024
    </div>
    """,
    unsafe_allow_html=True
)
