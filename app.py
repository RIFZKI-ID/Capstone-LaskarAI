import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="AgroDetect: Aplikasi Identifikasi Hama dan Penyakit Daun",
    page_icon="🌿",
    layout="wide"
)

# --- Konfigurasi CSS Kustom ---
st.markdown("""
<style>
    .stApp {
        /* background-color: #f9f9f9; */ /* Latar belakang sedikit abu-abu, opsional */
    }
    /* Tombol Utama */
    .stButton>button {
        background-color: #2E8B57; /* Hijau Hutan yang lebih lembut */
        color: white;
        border-radius: 8px; /* Sedikit lebih bulat */
        padding: 10px 20px;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1); /* Sedikit shadow */
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #256d42; /* Hijau lebih gelap saat hover */
        color: white;
    }
    /* Tombol Bersihkan (jika ingin dibedakan) */
    .stButton button[kind="secondary"] { /* Target tombol sekunder jika ada */
        background-color: #f0f2f6;
        color: #555;
    }
    .stButton button[kind="secondary"]:hover {
        background-color: #e0e2e6;
        color: #333;
    }

    .title-app {
        color: #2E8B57; /* Hijau Hutan */
        text-align: center;
        font-weight: bold;
        margin-bottom: 0.5rem; /* Mengurangi margin bawah */
    }
    .subtitle-app {
        text-align: center;
        color: #4A4A4A; /* Abu-abu gelap untuk subjudul */
        margin-bottom: 2rem;
    }
    .expander-custom {
        border: 1px solid #e0e0e0; /* Border lebih lembut */
        border-radius: 8px;
        padding: 15px;
        background-color: #ffffff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05); /* Shadow sangat halus */
        margin-bottom: 1rem;
    }
    .sidebar-title {
        color: #2E8B57;
        font-size: 1.8em;
        font-weight: bold;
        text-align: left;
    }
    .stFileUploader label {
        font-weight: bold;
        color: #333;
    }
    .stImage > img {
        border-radius: 8px; /* Rounded corner untuk gambar */
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


# --- Configuration ---
MODEL_PATH = "best_model.keras"
CONFIDENCE_THRESHOLD = 80


# --- Load the Model ---
@st.cache_resource
def load_ml_model():
    """Memuat model Keras dari path yang ditentukan."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading the ML model from '{MODEL_PATH}': {e}")
        st.warning(
            "Pastikan file model 'best_model.keras' berada di direktori yang sama dengan skrip, atau berikan path yang benar."
        )
        st.stop()


model = load_ml_model()

# --- Define Class Names ---
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
    """Memproses gambar yang diunggah agar sesuai untuk prediksi model."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    target_size = (128, 128)
    image = image.resize(target_size)
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# --- Sidebar ---
with st.sidebar:
    st.image("https://emojigraph.org/media/apple/leafy-green_1f96c.png", width=80, use_column_width='auto')
    st.markdown("<h2 class='sidebar-title'>AgroDetect</h2>", unsafe_allow_html=True)
    st.caption("Identifikasi Cepat, Pertanian Hebat!")

    st.markdown("---")
    st.subheader("Menu Navigasi")
    page_selection = st.radio("Pilih Halaman:", ["🏡 Identifikasi Tanaman", "ℹ️ Tentang AgroDetect", "👥 Detail Proyek"], label_visibility="collapsed")
    st.markdown("---")
    st.info("Aplikasi ini menggunakan model Machine Learning untuk mengidentifikasi penyakit pada daun tanaman.")


# --- Konten Halaman ---

if page_selection == "🏡 Identifikasi Tanaman":
    st.markdown("<h1 class='title-app'>🌿 AgroDetect</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='subtitle-app'>Aplikasi Identifikasi Hama dan Penyakit Daun Paprika, Tomat, dan Kentang</p>",
        unsafe_allow_html=True
    )

    st.header("🔍 Unggah Gambar untuk Identifikasi")

    uploaded_file = st.file_uploader(
        "Pilih gambar daun tanaman:",
        type=["jpg", "jpeg", "png"],
        help="Pastikan gambar jelas dan fokus pada daun yang terindikasi.",
        label_visibility="collapsed" # Menyembunyikan label default agar lebih bersih
    )

    col1, col2 = st.columns([0.6, 0.4]) # Kolom untuk gambar dan hasil

    with col1:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar yang Diunggah", use_container_width=True)
        else:
            st.info("Silakan unggah gambar daun untuk memulai identifikasi.")


    with col2:
        if uploaded_file is not None:
            if st.button("🚀 Lakukan Identifikasi", key="identify_button", help="Klik untuk memulai proses identifikasi", use_container_width=True):
                with st.spinner("⏳ Menganalisis gambar..."):
                    try:
                        processed_image = preprocess_image(image)
                        predictions = model.predict(processed_image)
                        predicted_class_index = np.argmax(predictions, axis=1)[0]
                        confidence = predictions[0][predicted_class_index] * 100

                        st.subheader("📊 Hasil Identifikasi:")

                        if confidence >= CONFIDENCE_THRESHOLD:
                            predicted_class_name = CLASS_NAMES[predicted_class_index]
                            st.success(f"**Identifikasi:** {predicted_class_name}", icon="✅")
                            st.metric(label="Keyakinan Model", value=f"{confidence:.2f}%")
                            st.info(
                                "Klasifikasi ini didasarkan pada analisis fitur visual dari foto yang diunggah, "
                                "seperti pola bercak dan perubahan warna, untuk identifikasi dini yang akurat dan cepat.",
                                icon="💡"
                            )
                        else:
                            st.warning(
                                "**Identifikasi:** Tidak dapat mengidentifikasi penyakit/hama dengan pasti.", icon="⚠️"
                            )
                            st.metric(label="Keyakinan Model (Tertinggi)", value=f"{confidence:.2f}%", delta=f"Di bawah {CONFIDENCE_THRESHOLD}%", delta_color="inverse")
                            st.info(
                                "Model tidak cukup yakin. Coba unggah gambar yang lebih jelas atau dari sudut yang berbeda. "
                                "Pastikan gambar fokus pada area daun yang menunjukkan gejala.",
                                icon="↪️"
                            )

                        st.markdown("---")
                        with st.expander(
                            "Lihat Detail Probabilitas per Kelas", expanded=False
                        ):
                            sorted_indices = np.argsort(predictions[0])[::-1]
                            for i in sorted_indices:
                                prob = predictions[0][i] * 100
                                class_name = CLASS_NAMES[i]
                                if (i == predicted_class_index and confidence >= CONFIDENCE_THRESHOLD):
                                    st.markdown(f"- **{class_name}: {prob:.2f}% (Teridentifikasi)**")
                                elif (i == predicted_class_index and confidence < CONFIDENCE_THRESHOLD):
                                    st.markdown(f"- *{class_name}: {prob:.2f}% (Keyakinan tertinggi)*")
                                else:
                                    st.write(f"- {class_name}: {prob:.2f}%")

                    except AttributeError as ae:
                        st.error(f"Terjadi kesalahan: Model tidak berhasil dimuat. Detail: {ae}", icon="❌")
                        st.info("Pastikan file model 'best_model.keras' ada dan tidak rusak.")
                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat memproses gambar atau memprediksi: {e}", icon="❌")
                        st.info("Pastikan gambar yang diunggah sesuai dan model Anda sudah dilatih dengan benar.")
            
            if st.button("🧹 Bersihkan", help="Klik untuk menghapus gambar dan hasil", use_container_width=True, type="secondary"):
                 st.rerun() # Ini akan mereset state termasuk uploaded_file

elif page_selection == "ℹ️ Tentang AgroDetect":
    st.markdown("<h1 class='title-app'>💡 Tentang AgroDetect</h1>", unsafe_allow_html=True)
    st.markdown(
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
        """,
        unsafe_allow_html=True,
    )

elif page_selection == "👥 Detail Proyek":
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
        <strong>Nama Advisor:</strong> Stevani Dwi Utomo, [Sesi mentoring dilakukan pada <em>(harap isi tanggal mentoring Anda)</em>]
        <br><br>
        <strong>Anggota Grup:</strong>
        <ul>
            <li>A180YBF358 – Mukhamad Ikhsanudin – Universitas Airlangga</li>
            <li>A706YBF391 – Patuh Rujhan Al Istizhar – Universitas Swadaya Gunung Jati</li>
            <li>A573YBF408 – Rahmat Hidayat – Universitas Lancang Kuning</li>
            <li>A314YBF428 – Rifzki Adiyaksa – Universitas Singaperbangsa Karawang</li>
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
    <div style="text-align: center; color: #777; font-size: 0.9em;">
        © 2024 Tim Capstone Laskar AI (LAI25-RM097)
    </div>
    <br>
    """,
    unsafe_allow_html=True
)
