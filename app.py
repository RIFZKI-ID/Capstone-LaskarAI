import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="AgroDetect: Asisten Kebun Cerdas",
    page_icon="🌱",
    layout="wide",  # Diubah dari 'centered' menjadi 'wide'
    initial_sidebar_state="expanded",
)

# --- Path Model ML & Threshold Keyakinan ---
MODEL_PATH = "best_model.keras"  # Pastikan file model ini ada di direktori yang sama
CONFIDENCE_THRESHOLD = 75  # Threshold keyakinan untuk hasil yang "dikonfirmasi"

# --- Data Penyakit & Informasi ---
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

# Kamus informasi detail penyakit/kesehatan (dengan gejala dan solusi yang lebih ringkas/poin-poin)
disease_info = {
    "Pepper_bell__Bacterial_spot": {
        "nama_display": "Bercak Bakteri Paprika",
        "deskripsi_singkat": "Bakteri menyebabkan bercak berminyak dan luka pada daun serta buah.",
        "penyebab": "Bakteri _Xanthomonas campestris_. Menyebar via air, angin, alat.",
        "gejala": [
            "Bercak kecil gelap berminyak dengan halo kuning pada daun.",
            "Luka berkerak pada buah.",
        ],
        "solusi": [
            "Benih sehat.",
            "Rotasi tanaman.",
            "Sanitasi kebun.",
            "Hindari penyiraman dari atas.",
            "Bakterisida tembaga.",
        ],
    },
    "Pepper_bell__healthy": {
        "nama_display": "Paprika Sehat",
        "deskripsi_singkat": "Tanaman paprika Anda dalam kondisi prima.",
        "penyebab": "Praktik budidaya yang baik.",
        "gejala": ["Daun hijau cerah, kuat.", "Tidak ada bercak atau perubahan warna."],
        "solusi": ["Pertahankan perawatan rutin.", "Pantau terus kesehatan tanaman."],
    },
    "Potato_Early_blight": {
        "nama_display": "Bercak Kering Kentang",
        "deskripsi_singkat": "Jamur _Alternaria solani_ menyebabkan bercak konsentris pada daun.",
        "penyebab": "Jamur _Alternaria solani_. Bertahan di sisa tanaman.",
        "gejala": [
            "Bercak bulat cokelat dengan pola cincin (target-like) pada daun tua."
        ],
        "solusi": [
            "Varietas tahan.",
            "Rotasi tanaman.",
            "Musnahkan sisa tanaman.",
            "Fungisida.",
        ],
    },
    "Potato_Late_blight": {
        "nama_display": "Busuk Daun Kentang",
        "deskripsi_singkat": "Penyakit jamur cepat menyebar yang merusak daun dan umbi.",
        "penyebab": "Jamur _Phytophthora infestans_. Berkembang pada suhu sejuk, lembap tinggi.",
        "gejala": [
            "Bercak basah gelap pada daun & batang.",
            "Kapang putih di bawah daun.",
            "Pembusukan umbi.",
        ],
        "solusi": [
            "Bibit sehat.",
            "Varietas tahan.",
            "Sirkulasi udara baik.",
            "Hindari penyiraman atas.",
            "Fungisida sistemik/kontak.",
        ],
    },
    "Potato_healthy": {
        "nama_display": "Kentang Sehat",
        "deskripsi_singkat": "Tanaman kentang Anda tumbuh dengan baik dan bebas penyakit.",
        "penyebab": "Lingkungan optimal, manajemen yang tepat.",
        "gejala": [
            "Daun hijau gelap, pertumbuhan kuat.",
            "Tidak ada tanda penyakit/hama.",
        ],
        "solusi": ["Lanjutkan perawatan rutin.", "Pantau dan jaga kebersihan lahan."],
    },
    "Tomato_Bacterial_spot": {
        "nama_display": "Bercak Bakteri Tomat",
        "deskripsi_singkat": "Bakteri menyebabkan bercak pada daun, batang, dan buah tomat.",
        "penyebab": "Bakteri _Xanthomonas_. Menyebar melalui percikan air, benih.",
        "gejala": [
            "Bercak kecil berair, gelap pada daun.",
            "Kerak menonjol pada buah.",
        ],
        "solusi": [
            "Benih/bibit bebas penyakit.",
            "Sanitasi.",
            "Rotasi tanaman.",
            "Hindari membasahi daun.",
            "Bakterisida tembaga.",
        ],
    },
    "Tomato_Early_blight": {
        "nama_display": "Bercak Kering Tomat",
        "deskripsi_singkat": "Jamur _Alternaria solani_ menyebabkan bercak dengan cincin konsentris.",
        "penyebab": "Jamur _Alternaria solani_. Bertahan di sisa tanaman.",
        "gejala": ["Bercak bulat cokelat dengan cincin konsentris pada daun tua."],
        "solusi": [
            "Varietas tahan.",
            "Rotasi tanaman.",
            "Bersihkan sisa tanaman.",
            "Fungisida.",
        ],
    },
    "Tomato_Late_blight": {
        "nama_display": "Busuk Daun Tomat",
        "deskripsi_singkat": "Penyakit jamur yang cepat menyebar, merusak seluruh bagian tanaman.",
        "penyebab": "Jamur _Phytophthora infestans_. Menyukai suhu sejuk, lembap tinggi.",
        "gejala": [
            "Bercak besar basah gelap.",
            "Kapang putih di bawah daun.",
            "Buah membusuk.",
        ],
        "solusi": [
            "Varietas tahan.",
            "Bibit sehat.",
            "Jaga sirkulasi udara.",
            "Hindari penyiraman atas.",
            "Fungisida.",
        ],
    },
    "Tomato_Leaf_Mold": {
        "nama_display": "Embun Tepung Tomat",
        "deskripsi_singkat": "Jamur menyebabkan lapisan seperti beludru di bawah daun, terutama di lingkungan lembap.",
        "penyebab": "Jamur _Passalora fulva_. Kondisi lembap tinggi (>85%), suhu sedang.",
        "gejala": [
            "Bercak kuning kehijauan di atas daun.",
            "Lapisan beludru cokelat keabu-abuan di bawah daun.",
        ],
        "solusi": [
            "Varietas resisten.",
            "Tingkatkan sirkulasi udara.",
            "Turunkan kelembaban.",
            "Hindari membasahi daun.",
            "Fungisida.",
        ],
    },
    "Tomato_Septoria_leaf_spot": {
        "nama_display": "Bercak Daun Septoria Tomat",
        "deskripsi_singkat": "Jamur menyebabkan bercak kecil bulat dengan titik hitam di tengah.",
        "penyebab": "Jamur _Septoria lycopersici_. Menyebar melalui percikan air.",
        "gejala": [
            "Bercak kecil bulat cokelat dengan pusat abu-abu dan titik hitam kecil (piknidia)."
        ],
        "solusi": [
            "Rotasi tanaman.",
            "Sanitasi kebun.",
            "Gunakan mulsa.",
            "Hindari penyiraman atas.",
            "Fungisida.",
        ],
    },
    "Tomato_Spider_mites_Two_spotted_mite": {
        "nama_display": "Tungau Laba-laba",
        "deskripsi_singkat": "Hama kecil penghisap cairan yang menyebabkan bintik kuning dan jaring halus.",
        "penyebab": "Tungau _Tetranychus urticae_. Berkembang biak cepat di kondisi panas, kering.",
        "gejala": [
            "Bintik kuning/perunggu pada daun.",
            "Jaring halus di antara daun.",
            "Daun menggulung/kering.",
        ],
        "solusi": [
            "Jaga kelembaban.",
            "Semprot air bertekanan.",
            "Musuh alami.",
            "Sabun insektisida/minyak nimba.",
            "Akarisida.",
        ],
    },
    "Tomato_Target_Spot": {
        "nama_display": "Bercak Sasaran Tomat",
        "deskripsi_singkat": "Jamur menyebabkan bercak konsentris seperti 'target' pada daun.",
        "penyebab": "Jamur _Corynespora cassiicola_. Menyebar via angin/air.",
        "gejala": ["Bercak bulat cokelat gelap dengan zona konsentris seperti target."],
        "solusi": [
            "Rotasi tanaman.",
            "Sanitasi.",
            "Drainase baik.",
            "Tingkatkan sirkulasi udara.",
            "Fungisida.",
        ],
    },
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus": {
        "nama_display": "Virus Kuning Keriting Daun Tomat (TYLCV)",
        "deskripsi_singkat": "Virus yang ditularkan kutu kebul, menyebabkan daun menguning dan keriting parah.",
        "penyebab": "Virus TYLCV ditularkan oleh kutu kebul (_Bemisia tabaci_).",
        "gejala": [
            "Daun menguning di antara tulang daun, menggulung ke atas.",
            "Tanaman kerdil, buah sedikit.",
        ],
        "solusi": [
            "Varietas tahan.",
            "Kendali kutu kebul.",
            "Mulsa perak.",
            "Sanitasi.",
            "Cabut tanaman terinfeksi.",
        ],
    },
    "Tomato_Tomato_mosaic_virus": {
        "nama_display": "Virus Mosaik Tomat (ToMV)",
        "deskripsi_singkat": "Virus sangat menular yang menyebabkan pola mosaik pada daun dan pertumbuhan terhambat.",
        "penyebab": "Virus ToMV. Mudah menular secara mekanis (sentuhan, alat, benih).",
        "gejala": [
            "Pola mosaik (hijau terang/gelap) pada daun.",
            "Daun keriting/cacat.",
            "Tanaman kerdil.",
        ],
        "solusi": [
            "Benih sehat.",
            "Cuci tangan & sterilkan alat.",
            "Hindari produk tembakau.",
            "Cabut tanaman terinfeksi.",
        ],
    },
    "Tomato_healthy": {
        "nama_display": "Tomat Sehat",
        "deskripsi_singkat": "Tanaman tomat Anda dalam kondisi prima dan produktif.",
        "penyebab": "Praktik budidaya yang optimal.",
        "gejala": [
            "Daun hijau gelap, pertumbuhan tegak.",
            "Tidak ada tanda penyakit/hama.",
        ],
        "solusi": [
            "Pertahankan perawatan.",
            "Inspeksi rutin.",
            "Optimalkan sirkulasi udara.",
            "Pemangkasan teratur.",
        ],
    },
}


# --- Fungsi Praproses Gambar ---
@st.cache_data
def preprocess_image(_image: Image.Image) -> np.ndarray:
    """Memproses gambar yang diunggah untuk prediksi model."""
    if _image.mode != "RGB":
        _image = _image.convert("RGB")
    target_size = (128, 128)
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
    except Exception:
        st.error(
            f"❌ **Ups!** Model Machine Learning gagal dimuat dari '{MODEL_PATH}'."
        )
        st.warning(
            "Ini mungkin terjadi jika file model tidak ada atau rusak. "
            "Pastikan `best_model.keras` berada di lokasi yang benar."
        )
        st.stop()


model = load_ml_model()


# --- Fungsi Reset State Aplikasi ---
def reset_app_state():
    """Meriset semua status sesi yang relevan untuk menghapus hasil dan memungkinkan unggahan baru."""
    st.session_state.uploaded_file = None  # Pastikan ini diset ke None
    st.session_state.identification_done = False
    st.session_state.predicted_class_name_state = None
    st.session_state.confidence_state = None
    st.session_state.predictions_state = None
    st.session_state.show_detailed_solution = False
    st.session_state.current_page = "Identifikasi"
    # Tambahkan kunci unik untuk file_uploader agar terreset sepenuhnya
    st.session_state.file_uploader_key = str(np.random.rand())


# --- Inisialisasi State Awal (PENTING: Semua variabel session_state harus diinisialisasi di sini) ---
if "identification_done" not in st.session_state:
    st.session_state.identification_done = False
    st.session_state.predicted_class_name_state = None
    st.session_state.confidence_state = None
    st.session_state.predictions_state = None
    st.session_state.show_detailed_solution = False
    st.session_state.current_page = "Identifikasi"
    st.session_state.file_uploader_key = "initial"  # Kunci awal untuk uploader
    st.session_state.uploaded_file = (
        None  # <<--- INI YANG DITAMBAHKAN/DIPERBAIKI UNTUK MENGATASI AttributeError
    )

# --- SIDEBAR NAVIGASI ---
with st.sidebar:
    st.image("https://emojigraph.org/media/apple/leafy-green_1f96c.png", width=80)
    st.markdown("## 🌱 **AgroDetect**")
    st.caption("_Asisten Kebun Cerdas Anda_")

    st.divider()

    if st.button(
        "🏡 Identifikasi Tanaman",
        use_container_width=True,
        type="primary"
        if st.session_state.current_page == "Identifikasi"
        else "secondary",
    ):
        st.session_state.current_page = "Identifikasi"
        reset_app_state()
        st.rerun()  # Penting untuk me-rerun agar uploader key di-apply

    if st.button(
        "💡 Tentang AgroDetect",
        use_container_width=True,
        type="primary" if st.session_state.current_page == "Tentang" else "secondary",
    ):
        st.session_state.current_page = "Tentang"

    if st.button(
        "👥 Tim Pengembang",
        use_container_width=True,
        type="primary" if st.session_state.current_page == "Tim" else "secondary",
    ):
        st.session_state.current_page = "Tim"

    st.divider()
    st.info("AgroDetect: Mempermudah petani mendeteksi penyakit dan hama dengan AI.")
    st.caption("© 2025 Laskar AI Capstone")

# --- KONTEN HALAMAN UTAMA BERDASARKAN NAVIGASI ---

if st.session_state.current_page == "Identifikasi":
    # --- BAGIAN HERO ---
    st.markdown(
        "<h1 style='text-align: center; color: #4CAF50;'>🌱 AgroDetect: Temukan Masalah Tanaman Anda!</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; font-size: 1.1em;'>Unggah gambar daun tanaman paprika, tomat, atau kentang Anda. "
        "Kami akan menganalisisnya dan memberikan diagnosis cepat serta rekomendasi penanganan.</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # --- AREA UNGGAH GAMBAR ---
    st.subheader("📸 Unggah Foto Daun")
    st.write("Seret & lepas gambar di sini, atau klik untuk memilih file.")

    # Menggunakan session_state.uploaded_file untuk mengontrol st.file_uploader
    # dan key unik untuk memaksa reset
    current_uploaded_file = st.file_uploader(
        "Pilih gambar daun (JPG, PNG):",
        type=["jpg", "jpeg", "png"],
        key=st.session_state.file_uploader_key,  # Menggunakan kunci unik
        label_visibility="collapsed",
    )

    # Logika untuk memperbarui st.session_state.uploaded_file berdasarkan current_uploaded_file
    if current_uploaded_file is not None:
        # Jika ada file baru diunggah, simpan ke session_state dan reset identifikasi
        if st.session_state.uploaded_file != current_uploaded_file:
            st.session_state.uploaded_file = current_uploaded_file
            st.session_state.identification_done = False
            st.session_state.predicted_class_name_state = None
            st.session_state.confidence_state = None
            st.session_state.predictions_state = None
            st.session_state.show_detailed_solution = False
    elif st.session_state.uploaded_file is not None and current_uploaded_file is None:
        # Ini terjadi jika file_uploader direset oleh key, atau pengguna menghapus file secara manual
        st.session_state.uploaded_file = None
        st.session_state.identification_done = False
        st.session_state.predicted_class_name_state = None
        st.session_state.confidence_state = None
        st.session_state.predictions_state = None
        st.session_state.show_detailed_solution = False

    # Tampilan Pratinjau Gambar (besar se-kontainer)
    if st.session_state.uploaded_file is not None:
        image = Image.open(st.session_state.uploaded_file)
        st.image(image, caption="Foto Daun Anda", use_container_width=True)
    else:
        st.markdown(
            "<div style='border: 2px dashed #4CAF50; padding: 50px; text-align: center; opacity: 0.7;'>"
            "Tidak ada gambar diunggah."
            "</div>",
            unsafe_allow_html=True,
        )
        st.info(
            "Unggah foto daun yang jelas agar hasil identifikasi lebih akurat. Fokus pada area yang menunjukkan gejala."
        )

    # Tombol Analisis di bawah pratinjau gambar
    if st.session_state.uploaded_file is not None:
        if st.button(
            "✨ **Mulai Analisis Cerdas!**",
            key="analyze_button",
            help="Klik untuk mengidentifikasi penyakit pada foto daun Anda.",
            use_container_width=True,
            type="primary",
        ):
            with st.spinner("⏳ Analisis sedang berlangsung..."):
                try:
                    processed_image = preprocess_image(
                        Image.open(st.session_state.uploaded_file)
                    )
                    predictions = model.predict(processed_image)
                    predicted_class_index = np.argmax(predictions, axis=1)[0]
                    confidence = predictions[0][predicted_class_index] * 100

                    st.session_state.predictions_state = predictions
                    st.session_state.confidence_state = confidence
                    st.session_state.identification_done = True
                    st.session_state.predicted_class_name_state = CLASS_NAMES[
                        predicted_class_index
                    ]

                except Exception as e:
                    st.error(
                        f"❌ **Terjadi kesalahan saat analisis:** {e}"
                        "Mohon coba lagi atau unggah gambar lain."
                    )
                    st.session_state.identification_done = False

        # --- TAMPILAN HASIL IDENTIFIKASI ---
        if st.session_state.identification_done:
            st.markdown("---")
            st.subheader("💡 Hasil Identifikasi")

            confidence = st.session_state.confidence_state
            predicted_class_name = st.session_state.predicted_class_name_state
            predictions = st.session_state.predictions_state

            info = disease_info.get(predicted_class_name, {})
            display_name = info.get(
                "nama_display",
                predicted_class_name.replace("_", " ").replace("__", ": "),
            )
            deskripsi_singkat = info.get(
                "deskripsi_singkat", "Informasi tambahan tidak tersedia."
            )

            if (
                confidence >= CONFIDENCE_THRESHOLD
                and "healthy" not in predicted_class_name.lower()
            ):
                st.markdown(
                    f"<h3 style='color: #E64A19;'>🚨 Terdeteksi: {display_name}</h3>",
                    unsafe_allow_html=True,
                )
                st.metric(
                    label="Tingkat Keyakinan",
                    value=f"{confidence:.2f}%",
                    delta="Penyakit Terdeteksi",
                    delta_color="inverse",
                )
                st.markdown(f"**Ringkasan:** {deskripsi_singkat}")

                if st.button(
                    "📖 Lihat Detail Solusi & Penanganan",
                    key="view_solution_button",
                    use_container_width=True,
                ):
                    st.session_state.show_detailed_solution = True

            elif "healthy" in predicted_class_name.lower():
                st.markdown(
                    f"<h3 style='color: #4CAF50;'>✅ Tanaman Sehat: {display_name}</h3>",
                    unsafe_allow_html=True,
                )
                st.metric(
                    label="Tingkat Keyakinan",
                    value=f"{confidence:.2f}%",
                    delta="Sehat",
                    delta_color="normal",
                )
                st.markdown(f"**Ringkasan:** {deskripsi_singkat}")

                if st.button(
                    "💚 Tips Menjaga Kesehatan Tanaman",
                    key="view_healthy_tips_button",
                    use_container_width=True,
                ):
                    st.session_state.show_detailed_solution = True

            else:
                st.markdown(
                    "<h3 style='color: #FFC107;'>❓ Hasil Kurang Yakin</h3>",
                    unsafe_allow_html=True,
                )
                st.metric(
                    label="Keyakinan Tertinggi",
                    value=f"{confidence:.2f}%",
                    delta=f"Di bawah {CONFIDENCE_THRESHOLD}%",
                    delta_color="off",
                )
                st.warning(
                    "Model belum bisa mengidentifikasi penyakit/hama dengan keyakinan tinggi. "
                    "Ini bisa karena gambar kurang jelas, atau penyakit yang tidak ada di dataset pelatihan kami. "
                    "Mohon coba unggah gambar lain yang lebih fokus pada gejala atau konsultasi dengan ahli."
                )

            # Tombol untuk memulai ulang
            st.markdown("---")
            if st.button(
                "🔄 **Mulai Unggah Gambar Baru**",
                help="Klik untuk menghapus hasil saat ini dan unggah foto daun yang lain.",
                use_container_width=True,
                type="secondary",
            ):
                reset_app_state()
                st.rerun()

    # --- BAGIAN DETAIL SOLUSI (DITAMPILKAN SECARA KONDISIONAL) ---
    if (
        st.session_state.get("show_detailed_solution", False)
        and st.session_state.identification_done
    ):
        st.markdown("---")
        st.header(f"🌿 Detail Penanganan untuk {display_name}")

        info_detail = disease_info.get(st.session_state.predicted_class_name_state, {})

        if info_detail:
            col_detail_1, col_detail_2 = st.columns(2)

            with col_detail_1:
                with st.expander("📚 **Penyebab & Gejala Khas**", expanded=True):
                    st.markdown(
                        f"**Penyebab Utama:** {info_detail.get('penyebab', 'Tidak tersedia.')}"
                    )
                    st.markdown("**Gejala yang Perlu Diperhatikan:**")
                    if isinstance(info_detail.get("gejala"), list):
                        for g in info_detail["gejala"]:
                            st.markdown(f"- {g}")
                    else:
                        st.write(info_detail.get("gejala", "Tidak tersedia."))

            with col_detail_2:
                with st.expander("👨‍🌾 **Langkah Solusi & Penanganan**", expanded=True):
                    st.markdown("**Rekomendasi:**")
                    if isinstance(info_detail.get("solusi"), list):
                        for s in info_detail["solusi"]:
                            st.markdown(f"- {s}")
                    else:
                        st.write(info_detail.get("solusi", "Tidak tersedia."))

            st.divider()
            with st.expander("🔬 **Probabilitas Lengkap (Untuk Ahli)**"):
                st.write(
                    "Berikut adalah daftar probabilitas model untuk setiap kategori, dari tertinggi ke terendah:"
                )
                sorted_indices = np.argsort(predictions[0])[::-1]
                for i in sorted_indices:
                    prob = predictions[0][i] * 100
                    class_disp = disease_info.get(CLASS_NAMES[i], {}).get(
                        "nama_display",
                        CLASS_NAMES[i].replace("_", " ").replace("__", ": "),
                    )
                    if i == np.argmax(predictions, axis=1)[0]:
                        st.markdown(f"- **{class_disp}: {prob:.2f}%** (Prediksi Utama)")
                    else:
                        st.write(f"- {class_disp}: {prob:.2f}%")
        else:
            st.warning(
                "Maaf, detail informasi untuk hasil ini tidak tersedia dalam database kami."
            )

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
    st.info(
        "Aplikasi ini adalah alat bantu diagnosa awal dan tidak menggantikan konsultasi dengan ahli pertanian profesional."
    )

elif st.session_state.current_page == "Tim":
    st.title("👨‍💻 Tim Pengembang")
    st.write("AgroDetect adalah hasil dari proyek Capstone oleh **Tim Laskar AI**.")
    st.divider()

    st.subheader("Informasi Proyek")
    st.markdown(
        """
        -   **ID Grup:** LAI25-RM097
        -   **Tema:** Solusi Cerdas untuk Masa Depan yang Lebih Baik
        -   **Pembimbing:** Stevani Dwi Utomo (Sesi mentoring: 5 Juni 2025)
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
st.markdown(
    "<p style='text-align: center; color: grey;'>© 2025 AgroDetect. Hak cipta dilindungi.</p>",
    unsafe_allow_html=True,
)
