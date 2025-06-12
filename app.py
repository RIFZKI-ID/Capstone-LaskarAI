import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="AgroDetect: Asisten Kebun Cerdas",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Path Model ML & Ambang Batas ---
MODEL_PATH = "best_model.keras"
# Tingkat keyakinan minimum untuk prediksi utama
CONFIDENCE_THRESHOLD = 80
# PERBAIKAN: Jarak minimum antara prediksi teratas dan kedua untuk memastikan model tidak "bingung"
CERTAINTY_GAP_THRESHOLD = 25 

# --- Data Penyakit & Informasi (Tetap sama) ---
CLASS_NAMES = [
    "Pepper_bell__Bacterial_spot", "Pepper_bell__healthy", "Potato_Early_blight",
    "Potato_Late_blight", "Potato_healthy", "Tomato_Bacterial_spot", "Tomato_Early_blight",
    "Tomato_Late_blight", "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_mite", "Tomato_Target_Spot",
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus", "Tomato_Tomato_mosaic_virus", "Tomato_healthy",
]

disease_info = {
    "Pepper_bell__Bacterial_spot": {
        "nama_tampilan": "Bercak Bakteri Paprika",
        "deskripsi_singkat": "Bakteri menyebabkan bercak berminyak dan luka pada daun serta buah.",
        "penyebab": "Bakteri _Xanthomonas campestris_. Menyebar via air, angin, alat.",
        "gejala": ["Bercak kecil gelap berminyak dengan halo kuning pada daun.", "Luka berkerak pada buah."],
        "solusi": ["Gunakan benih sehat.", "Rotasi tanaman.", "Sanitasi kebun.", "Hindari penyiraman dari atas.", "Bakterisida berbahan dasar tembaga."],
    },
    "Pepper_bell__healthy": {
        "nama_tampilan": "Paprika Sehat",
        "deskripsi_singkat": "Tanaman paprika Anda dalam kondisi prima.",
        "penyebab": "Praktik budidaya yang baik.",
        "gejala": ["Daun hijau cerah, kuat.", "Tidak ada bercak atau perubahan warna."],
        "solusi": ["Pertahankan perawatan rutin.", "Pantau terus kesehatan tanaman."],
    },
    "Potato_Early_blight": {
        "nama_tampilan": "Bercak Kering Kentang",
        "deskripsi_singkat": "Jamur _Alternaria solani_ menyebabkan bercak konsentris pada daun.",
        "penyebab": "Jamur _Alternaria solani_. Bertahan di sisa tanaman.",
        "gejala": ["Bercak bulat cokelat dengan pola cincin (seperti target) pada daun tua."],
        "solusi": ["Gunakan varietas tahan.", "Rotasi tanaman.", "Musnahkan sisa tanaman.", "Gunakan fungisida."],
    },
    "Potato_Late_blight": {
        "nama_tampilan": "Busuk Daun Kentang",
        "deskripsi_singkat": "Penyakit jamur cepat menyebar yang merusak daun dan umbi.",
        "penyebab": "Jamur _Phytophthora infestans_. Berkembang pada suhu sejuk, kelembapan tinggi.",
        "gejala": ["Bercak basah gelap pada daun & batang.", "Kapang putih di bawah daun.", "Pembusukan umbi."],
        "solusi": ["Gunakan bibit sehat.", "Gunakan varietas tahan.", "Pastikan sirkulasi udara baik.", "Hindari penyiraman dari atas.", "Gunakan fungisida sistemik/kontak."],
    },
    "Potato_healthy": {
        "nama_tampilan": "Kentang Sehat",
        "deskripsi_singkat": "Tanaman kentang Anda tumbuh dengan baik dan bebas penyakit.",
        "penyebab": "Lingkungan optimal, manajemen yang tepat.",
        "gejala": ["Daun hijau gelap, pertumbuhan kuat.", "Tidak ada tanda penyakit/hama."],
        "solusi": ["Lanjutkan perawatan rutin.", "Pantau dan jaga kebersihan lahan."],
    },
    "Tomato_Bacterial_spot": {
        "nama_tampilan": "Bercak Bakteri Tomat",
        "deskripsi_singkat": "Bakteri menyebabkan bercak pada daun, batang, dan buah tomat.",
        "penyebab": "Bakteri _Xanthomonas_. Menyebar melalui percikan air, benih.",
        "gejala": ["Bercak kecil berair, gelap pada daun.", "Kerak menonjol pada buah."],
        "solusi": ["Gunakan benih/bibit bebas penyakit.", "Jaga sanitasi.", "Lakukan rotasi tanaman.", "Hindari membasahi daun.", "Gunakan bakterisida berbahan dasar tembaga."],
    },
    "Tomato_Early_blight": {
        "nama_tampilan": "Bercak Kering Tomat",
        "deskripsi_singkat": "Jamur _Alternaria solani_ menyebabkan bercak dengan cincin konsentris.",
        "penyebab": "Jamur _Alternaria solani_. Bertahan di sisa tanaman.",
        "gejala": ["Bercak bulat cokelat dengan cincin konsentris pada daun tua."],
        "solusi": ["Gunakan varietas tahan.", "Lakukan rotasi tanaman.", "Bersihkan sisa tanaman.", "Gunakan fungisida."],
    },
    "Tomato_Late_blight": {
        "nama_tampilan": "Busuk Daun Tomat",
        "deskripsi_singkat": "Penyakit jamur yang cepat menyebar, merusak seluruh bagian tanaman.",
        "penyebab": "Jamur _Phytophthora infestans_. Menyukai suhu sejuk, kelembapan tinggi.",
        "gejala": ["Bercak besar basah gelap.", "Kapang putih di bawah daun.", "Buah membusuk."],
        "solusi": ["Gunakan varietas tahan.", "Gunakan bibit sehat.", "Jaga sirkulasi udara.", "Hindari penyiraman dari atas.", "Gunakan fungisida."],
    },
    "Tomato_Leaf_Mold": {
        "nama_tampilan": "Kapang Daun Tomat",
        "deskripsi_singkat": "Jamur menyebabkan lapisan seperti beludru di bawah daun, terutama di lingkungan lembap.",
        "penyebab": "Jamur _Passalora fulva_ (syn. _Fulvia fulva_). Kondisi lembap tinggi (>85%), suhu sedang.",
        "gejala": ["Bercak kuning kehijauan di atas daun.", "Lapisan beludru cokelat keabu-abuan di bawah daun."],
        "solusi": ["Gunakan varietas resisten.", "Tingkatkan sirkulasi udara.", "Turunkan kelembapan.", "Hindari membasahi daun.", "Gunakan fungisida."],
    },
    "Tomato_Septoria_leaf_spot": {
        "nama_tampilan": "Bercak Daun Septoria Tomat",
        "deskripsi_singkat": "Jamur menyebabkan bercak kecil bulat dengan titik hitam di tengah.",
        "penyebab": "Jamur _Septoria lycopersici_. Menyebar melalui percikan air.",
        "gejala": ["Bercak kecil bulat cokelat dengan pusat abu-abu dan titik hitam kecil (piknidia)."],
        "solusi": ["Lakukan rotasi tanaman.", "Jaga sanitasi kebun.", "Gunakan mulsa.", "Hindari penyiraman dari atas.", "Gunakan fungisida."],
    },
    "Tomato_Spider_mites_Two_spotted_mite": {
        "nama_tampilan": "Tungau Laba-laba Tomat (Tungau Bercak Dua)",
        "deskripsi_singkat": "Hama kecil penghisap cairan yang menyebabkan bintik kuning dan jaring halus.",
        "penyebab": "Tungau _Tetranychus urticae_. Berkembang biak cepat di kondisi panas, kering.",
        "gejala": ["Bintik kuning/perunggu pada daun.", "Jaring halus di antara daun.", "Daun menggulung/kering."],
        "solusi": ["Jaga kelembapan.", "Semprot dengan air bertekanan.", "Gunakan musuh alami (misalnya, tungau predator).", "Gunakan sabun insektisida/minyak nimba.", "Gunakan akarisida (mitisida)."],
    },
    "Tomato_Target_Spot": {
        "nama_tampilan": "Bercak Target Tomat",
        "deskripsi_singkat": "Jamur menyebabkan bercak konsentris seperti 'target' pada daun.",
        "penyebab": "Jamur _Corynespora cassiicola_. Menyebar via angin/air.",
        "gejala": ["Bercak bulat cokelat gelap dengan zona konsentris seperti target."],
        "solusi": ["Lakukan rotasi tanaman.", "Jaga sanitasi.", "Pastikan drainase baik.", "Tingkatkan sirkulasi udara.", "Gunakan fungisida."],
    },
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus": {
        "nama_tampilan": "Virus Keriting Kuning Daun Tomat (TYLCV)",
        "deskripsi_singkat": "Virus yang ditularkan kutu kebul, menyebabkan daun menguning dan keriting parah.",
        "penyebab": "Virus TYLCV ditularkan oleh kutu kebul (_Bemisia tabaci_).",
        "gejala": ["Daun menguning di antara tulang daun, menggulung ke atas.", "Tanaman kerdil, buah sedikit."],
        "solusi": ["Gunakan varietas tahan.", "Kendalikan kutu kebul.", "Gunakan mulsa perak.", "Jaga sanitasi.", "Cabut tanaman terinfeksi."],
    },
    "Tomato_Tomato_mosaic_virus": {
        "nama_tampilan": "Virus Mosaik Tomat (ToMV)",
        "deskripsi_singkat": "Virus sangat menular yang menyebabkan pola mosaik pada daun dan pertumbuhan terhambat.",
        "penyebab": "Virus ToMV. Mudah menular secara mekanis (sentuhan, alat, benih).",
        "gejala": ["Pola mosaik (hijau terang/gelap) pada daun.", "Daun keriting/cacat.", "Tanaman kerdil."],
        "solusi": ["Gunakan benih sehat.", "Cuci tangan & sterilkan alat.", "Hindari produk tembakau di sekitar tanaman.", "Cabut tanaman terinfeksi."],
    },
    "Tomato_healthy": {
        "nama_tampilan": "Tomat Sehat",
        "deskripsi_singkat": "Tanaman tomat Anda dalam kondisi prima dan produktif.",
        "penyebab": "Praktik budidaya yang optimal.",
        "gejala": ["Daun hijau gelap, pertumbuhan tegak.", "Tidak ada tanda penyakit/hama."],
        "solusi": ["Pertahankan perawatan.", "Lakukan inspeksi rutin.", "Optimalkan sirkulasi udara.", "Lakukan pemangkasan teratur."],
    },
}

# --- Fungsi Praproses Gambar ---
def preprocess_image(_image: Image.Image) -> np.ndarray:
    """Memproses gambar yang diunggah untuk prediksi model."""
    if _image.mode != "RGB":
        _image = _image.convert("RGB")
    target_size = (224, 224)
    _image = _image.resize(target_size)
    img_array = np.array(_image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- Cache Model ML ---
@st.cache_resource
def load_ml_model():
    """Memuat model TensorFlow/Keras yang sudah dilatih."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ **Ups!** Model Machine Learning gagal dimuat dari '{MODEL_PATH}'. Kesalahan: {e}")
        st.warning("Pastikan `best_model.keras` berada di lokasi yang benar.")
        st.stop()

model = load_ml_model()

# --- Fungsi Reset State Aplikasi ---
def reset_app_state():
    """Meriset semua status sesi yang relevan untuk menghapus hasil analisis."""
    st.session_state.identification_done = False
    st.session_state.predicted_class_name_state = None
    st.session_state.confidence_state = None
    st.session_state.predictions_state = None
    st.session_state.show_detailed_solution = False
    st.session_state.threshold_message = None

# --- Inisialisasi State Awal ---
if "current_page" not in st.session_state:
    st.session_state.current_page = "Identifikasi"
if "identification_done" not in st.session_state:
    reset_app_state()

# --- NAVIGASI SIDEBAR ---
with st.sidebar:
    st.image("logo/leafy-green_1f96c.png", width=80)
    st.markdown("## 🌱 **AgroDetect**")
    st.caption("_Asisten Kebun Cerdas Anda_")
    st.divider()

    if st.button("🏡 Identifikasi Tanaman", use_container_width=True, type="primary" if st.session_state.current_page == "Identifikasi" else "secondary"):
        st.session_state.current_page = "Identifikasi"
        st.rerun()

    if st.button("💡 Tentang AgroDetect", use_container_width=True, type="primary" if st.session_state.current_page == "Tentang" else "secondary"):
        st.session_state.current_page = "Tentang"
        st.rerun()

    if st.button("👥 Tim Pengembang", use_container_width=True, type="primary" if st.session_state.current_page == "Tim" else "secondary"):
        st.session_state.current_page = "Tim"
        st.rerun()

    st.divider()
    st.info("AgroDetect: Mempermudah petani mendeteksi penyakit dan hama dengan AI.")
    st.caption("© 2025 Laskar AI Capstone")

# --- KONTEN HALAMAN UTAMA BERDASARKAN NAVIGASI ---
if st.session_state.current_page == "Identifikasi":
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🌱 AgroDetect: Temukan Masalah Tanaman Anda!</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.1em;'>Unggah gambar daun tanaman paprika, tomat, atau kentang Anda. Kami akan menganalisisnya dan memberikan diagnosis cepat serta rekomendasi penanganan.</p>", unsafe_allow_html=True)
    st.divider()

    st.subheader("📸 Unggah Foto Daun")
    st.write("Seret & lepas gambar di sini, atau klik untuk memilih file.")
    
    uploaded_file = st.file_uploader(
        "Pilih gambar daun (JPG, PNG):",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        if st.session_state.identification_done:
            reset_app_state()
        st.markdown(
            "<div style='border: 2px dashed #4CAF50; padding: 50px; text-align: center; opacity: 0.7;'>"
            "Tidak ada gambar diunggah."
            "</div>",
            unsafe_allow_html=True,
        )
        st.info("Unggah foto daun yang jelas agar hasil identifikasi lebih akurat. Fokus pada area yang menunjukkan gejala.")

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Foto Daun Anda", use_container_width=True)

            if st.button(
                "✨ **Mulai Analisis Cerdas!**",
                key="analyze_button",
                help="Klik untuk mengidentifikasi penyakit pada foto daun Anda.",
                use_container_width=True,
                type="primary",
            ):
                reset_app_state() 

                with st.spinner("⏳ Analisis sedang berlangsung..."):
                    try:
                        processed_image = preprocess_image(image)
                        predictions = model.predict(processed_image)[0] # Ambil array probabilitas
                        
                        # --- LOGIKA VALIDASI DUA TINGKAT ---
                        # 1. Urutkan prediksi dari tertinggi ke terendah
                        sorted_indices = np.argsort(predictions)[::-1]
                        top_pred_index = sorted_indices[0]
                        top_pred_confidence = predictions[top_pred_index] * 100
                        
                        second_pred_confidence = predictions[sorted_indices[1]] * 100
                        
                        # 2. Hitung jarak keyakinan
                        confidence_gap = top_pred_confidence - second_pred_confidence

                        # 3. Terapkan validasi
                        is_confident = top_pred_confidence >= CONFIDENCE_THRESHOLD
                        is_certain = confidence_gap >= CERTAINTY_GAP_THRESHOLD

                        if not (is_confident and is_certain):
                            st.session_state.threshold_message = (
                                f"⚠️ **Tidak Dapat Diidentifikasi Secara Akurat!**\n\n"
                                f"Model tidak dapat mengenali gambar ini dengan keyakinan dan kepastian yang cukup.\n"
                                f"*(Keyakinan teratas: {top_pred_confidence:.2f}%, Jarak keyakinan: {confidence_gap:.2f}%)*\n\n"
                                f"**Kemungkinan Penyebab:**\n"
                                f"- Gambar bukan daun tomat, kentang, atau paprika.\n"
                                f"- Kualitas gambar kurang jelas (buram, terlalu gelap/terang).\n"
                                f"- Model ragu-ragu antara dua kemungkinan penyakit.\n\n"
                                f"Mohon unggah gambar daun yang lebih jelas dan fokus dari tanaman yang didukung."
                            )
                        else:
                            st.session_state.threshold_message = None
                            st.session_state.predictions_state = [predictions] # Simpan dalam list untuk konsistensi
                            st.session_state.confidence_state = top_pred_confidence
                            st.session_state.predicted_class_name_state = CLASS_NAMES[top_pred_index]
                        
                        st.session_state.identification_done = True

                    except Exception as e:
                        st.error(f"❌ **Terjadi kesalahan saat analisis:** {e}. Mohon coba lagi.")
                        st.session_state.identification_done = False
                
                st.rerun()

        except Exception as e:
            st.error(f"Gagal memuat atau menampilkan gambar: {e}. Pastikan file valid.")
            reset_app_state()
            st.rerun()
            
    # Bagian untuk menampilkan hasil identifikasi dan solusi
    if st.session_state.identification_done:
        st.markdown("---")
        st.subheader("💡 Hasil Identifikasi")

        if st.session_state.threshold_message:
            st.warning(st.session_state.threshold_message)
        else:
            confidence = st.session_state.confidence_state
            predicted_class_name = st.session_state.predicted_class_name_state
            info = disease_info.get(predicted_class_name, {})
            display_name = info.get("nama_tampilan", predicted_class_name.replace("_", " ").replace("__", ": "))
            brief_description = info.get("deskripsi_singkat", "Informasi tambahan tidak tersedia.")

            if "healthy" in predicted_class_name.lower():
                st.success(f"✅ Tanaman Sehat: {display_name}")
                st.metric(label="Tingkat Keyakinan", value=f"{confidence:.2f}%", delta="Sehat", delta_color="normal")
            else:
                st.error(f"🚨 Terdeteksi: {display_name}")
                st.metric(label="Tingkat Keyakinan", value=f"{confidence:.2f}%", delta="Penyakit Terdeteksi", delta_color="inverse")
            
            st.markdown(f"**Ringkasan:** {brief_description}")
            if st.button("📖 Lihat Detail Solusi & Penanganan", use_container_width=True):
                st.session_state.show_detailed_solution = True
                st.rerun()

        st.markdown("---")
        if st.button(
            "🔄 **Analisis Gambar Lain**",
            help="Klik untuk kembali ke halaman unggah.",
            use_container_width=True,
            type="secondary",
        ):
            reset_app_state()
            st.rerun()

    # Bagian untuk menampilkan solusi detail
    if st.session_state.get("show_detailed_solution", False) and st.session_state.identification_done and not st.session_state.threshold_message:
        st.markdown("---")
        current_display_name = disease_info.get(st.session_state.predicted_class_name_state, {}).get("nama_tampilan", st.session_state.predicted_class_name_state.replace("_", " ").replace("__", ": "))
        st.header(f"🌿 Penanganan Detail untuk {current_display_name}")
        info_detail = disease_info.get(st.session_state.predicted_class_name_state, {})
        if info_detail:
            col_detail_1, col_detail_2 = st.columns(2)
            with col_detail_1:
                with st.expander("📚 **Penyebab & Gejala Khas**", expanded=True):
                    st.markdown(f"**Penyebab Utama:** {info_detail.get('penyebab', 'Tidak tersedia.')}")
                    st.markdown("**Gejala yang Perlu Diperhatikan:**")
                    if isinstance(info_detail.get("gejala"), list):
                        for symptom in info_detail["gejala"]:
                            st.markdown(f"- {symptom}")
                    else:
                        st.write(info_detail.get("gejala", "Tidak tersedia."))
            with col_detail_2:
                with st.expander("👨‍🌾 **Langkah Solusi & Penanganan**", expanded=True):
                    st.markdown("**Rekomendasi:**")
                    if isinstance(info_detail.get("solusi"), list):
                        for solution_step in info_detail["solusi"]:
                            st.markdown(f"- {solution_step}")
                    else:
                        st.write(info_detail.get("solusi", "Tidak tersedia."))
            st.divider()
            if st.session_state.predictions_state is not None:
                with st.expander("🔬 **Probabilitas Lengkap (Untuk Ahli)**"):
                    st.write("Berikut adalah daftar probabilitas model untuk setiap kategori, dari tertinggi ke terendah:")
                    # Ambil kembali array probabilitas dari state
                    current_predictions = st.session_state.predictions_state[0]
                    sorted_indices = np.argsort(current_predictions)[::-1]
                    for i in sorted_indices:
                        prob = current_predictions[i] * 100
                        class_disp = disease_info.get(CLASS_NAMES[i], {}).get("nama_tampilan", CLASS_NAMES[i].replace("_", " ").replace("__", ": "))
                        if i == np.argmax(current_predictions):
                            st.markdown(f"- **{class_disp}: {prob:.2f}%** (Prediksi Utama)")
                        else:
                            st.write(f"- {class_disp}: {prob:.2f}%")
        else:
            st.warning("Maaf, informasi detail untuk hasil ini tidak tersedia dalam basis data kami.")

elif st.session_state.current_page == "Tentang":
    st.title("💡 Tentang AgroDetect")
    st.write(
        """
        **AgroDetect** adalah aplikasi web inovatif yang memberdayakan petani modern dengan kekuatan **Machine Learning**.
        Misi kami adalah memberikan kemampuan deteksi dini hama dan penyakit pada daun **paprika, tomat, dan kentang**
        hanya melalui unggahan foto.
        """
    )
    st.divider()
    st.subheader("Visi & Misi Kami")
    st.write(
        """
        **Visi:** Menjadi platform terdepan yang mendukung pertanian berkelanjutan melalui solusi AI cerdas.
        **Misi:** Menyediakan alat identifikasi penyakit tanaman yang akurat dan mudah diakses, serta rekomendasi penanganan praktis untuk meningkatkan produktivitas pertanian.
        """
    )
    st.subheader("Teknologi di Balik Layar")
    st.write(
        """
        AgroDetect dibangun di atas model **Convolutional Neural Network (CNN)** yang canggih, dilatih dengan dataset
        **[Plant Village](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)** yang masif dan beragam.
        Ini memungkinkan model kami untuk mengenali pola dan gejala spesifik berbagai kondisi tanaman.
        """
    )
    st.info("Aplikasi ini adalah alat bantu diagnosa awal dan tidak menggantikan konsultasi dengan ahli pertanian profesional.")

elif st.session_state.current_page == "Tim":
    st.title("👨‍💻 Tim Pengembang")
    st.write("AgroDetect adalah hasil dari proyek Capstone oleh **Tim Laskar AI**.")
    st.divider()
    st.subheader("Informasi Proyek")
    st.markdown(
        """
        -   **ID Grup:** LAI25-RM097
        -   **Tema:** Solusi Cerdas untuk Masa Depan yang Lebih Baik
        -   **Pembimbing:** Stevani Dwi Utomo
        """
    )
    st.subheader("Anggota Tim")
    st.markdown(
        """
        Kami adalah individu yang bersemangat dalam menerapkan AI untuk solusi nyata:
        -   **Mukhamad Ikhsanudin** (A180YBF358) – Universitas Airlangga
        -   **Patuh Rujhan Al Istizhar** (A706YBF391) – Universitas Swadaya Gunung Jati
        -   **Rahmat Hidayat** (A573YBF408) – Universitas Lancang Kuning
        -   **Rifzki Adiyaksa** (A314YBF428) – Universitas Singaperbangsa Karawang
        """
    )
    st.info("Bersama, kami menciptakan inovasi untuk pertanian yang lebih baik.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>© 2025 AgroDetect. Hak cipta dilindungi.</p>", unsafe_allow_html=True)
