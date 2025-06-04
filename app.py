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

# --- Load the Model ---
MODEL_PATH = "best_model.keras"
CONFIDENCE_THRESHOLD = 80

@st.cache_resource
def load_ml_model():
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
    "Pepper_bell__Bacterial_spot", "Pepper_bell__healthy", "Potato_Early_blight",
    "Potato_Late_blight", "Potato_healthy", "Tomato_Bacterial_spot",
    "Tomato_Early_blight", "Tomato_Late_blight", "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites_Two_spotted_mite",
    "Tomato_Target_Spot", "Tomato_Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_Tomato_mosaic_virus", "Tomato_healthy",
]

# --- Informasi Penyebab dan Solusi Penyakit (PASTIKAN INI DIISI LENGKAP) ---
disease_info = {
    "Pepper_bell__Bacterial_spot": {
        "penyebab": "Disebabkan oleh bakteri Xanthomonas campestris pv. vesicatoria. Penyebaran patogen ini dibantu oleh percikan air hujan, irigasi, angin, dan peralatan pertanian yang terkontaminasi. Kelembaban tinggi dan suhu hangat mempercepat perkembangan penyakit.",
        "solusi": [
            "Gunakan benih yang sehat dan bebas penyakit.",
            "Lakukan rotasi tanaman dengan tanaman bukan famili Solanaceae (terong-terongan).",
            "Jaga kebersihan kebun dengan memusnahkan sisa-sisa tanaman terinfeksi.",
            "Hindari penyiraman berlebihan yang menyebabkan daun basah dalam waktu lama.",
            "Semprot dengan bakterisida berbahan aktif tembaga atau antibiotik pertanian sesuai dosis anjuran jika serangan parah.",
            "Tingkatkan sirkulasi udara di sekitar tanaman."
        ]
    },
    "Potato_Early_blight": {
        "penyebab": "Disebabkan oleh jamur Alternaria solani. Jamur ini bertahan pada sisa-sisa tanaman sakit atau pada inang alternatif. Penyebaran spora jamur dibantu oleh angin dan percikan air. Kondisi lembab dan hangat (24-29°C) mendukung perkembangan penyakit.",
        "solusi": [
            "Tanam varietas kentang yang tahan.",
            "Lakukan rotasi tanaman dengan tanaman yang bukan inang jamur ini.",
            "Musnahkan sisa-sisa tanaman yang terinfeksi setelah panen.",
            "Pastikan drainase lahan baik untuk mengurangi kelembaban.",
            "Berikan jarak tanam yang cukup untuk sirkulasi udara yang baik.",
            "Aplikasikan fungisida protektan atau sistemik sesuai anjuran jika gejala mulai terlihat atau sebagai tindakan pencegahan di daerah endemik."
        ]
    },
    "Potato_Late_blight": {
        "penyebab": "Disebabkan oleh jamur Phytophthora infestans. Patogen ini berkembang pesat pada kondisi suhu sejuk (15-20°C) dan kelembaban tinggi (di atas 90%), terutama saat malam hari yang dingin diikuti siang hari yang hangat dan lembab. Spora menyebar melalui angin dan percikan air.",
        "solusi": [
            "Gunakan bibit kentang yang sehat dan bersertifikat.",
            "Tanam varietas yang memiliki ketahanan terhadap busuk daun.",
            "Jaga kebersihan lahan dari gulma dan sisa tanaman.",
            "Lakukan penimbunan pada pangkal batang untuk melindungi umbi.",
            "Hindari penyiraman daun, usahakan air langsung ke tanah.",
            "Aplikasikan fungisida secara preventif atau kuratif sesuai dengan anjuran, terutama saat kondisi cuaca mendukung perkembangan penyakit. Fungisida sistemik atau kontak dapat digunakan."
        ]
    },
    "Tomato_Bacterial_spot": {
        "penyebab": "Disebabkan oleh beberapa spesies bakteri Xanthomonas (misalnya Xanthomonas perforans, X. vesicatoria, X. euvesicatoria, atau X. gardneri). Bakteri ini menyebar melalui percikan air (hujan atau irigasi), benih yang terinfeksi, bibit, dan peralatan pertanian. Infeksi sering terjadi melalui luka pada tanaman atau stomata. Kelembaban tinggi dan suhu hangat adalah kondisi ideal.",
        "solusi": [
            "Gunakan benih dan bibit yang bebas penyakit.",
            "Lakukan sanitasi dengan membersihkan sisa-sisa tanaman terinfeksi.",
            "Rotasi tanaman dengan tanaman bukan famili Solanaceae.",
            "Hindari membasahi daun saat menyiram.",
            "Semprot dengan bakterisida berbahan aktif tembaga secara preventif, terutama saat cuaca lembab dan hangat. Jika sudah terjadi infeksi, aplikasi bakterisida mungkin kurang efektif.",
            "Hindari bekerja di lahan saat tanaman basah untuk mengurangi penyebaran."
        ]
    },
    "Tomato_Early_blight": {
        "penyebab": "Disebabkan oleh jamur Alternaria solani (kadang juga Alternaria tomatophila). Jamur ini bertahan pada sisa-sisa tanaman di tanah, benih, atau gulma inang. Spora disebarkan oleh angin, air, serangga, dan peralatan. Kondisi hangat, lembab, dan stres pada tanaman (misalnya kekurangan nutrisi) mendukung perkembangan penyakit.",
        "solusi": [
            "Tanam varietas tomat yang tahan.",
            "Rotasi tanaman minimal 2-3 tahun dengan tanaman bukan famili Solanaceae.",
            "Sanitasi kebun dengan membersihkan dan memusnahkan sisa tanaman terinfeksi.",
            "Berikan pupuk berimbang untuk menjaga kesehatan tanaman.",
            "Pastikan jarak tanam cukup untuk sirkulasi udara yang baik.",
            "Pangkas daun bagian bawah yang bersentuhan dengan tanah.",
            "Aplikasikan fungisida protektan (seperti mancozeb, chlorothalonil) secara berkala, terutama saat cuaca lembab. Fungisida sistemik bisa digunakan jika serangan sudah terjadi."
        ]
    },
    "Tomato_Late_blight": {
        "penyebab": "Disebabkan oleh jamur Phytophthora infestans, sama seperti pada kentang. Berkembang pada suhu sejuk dan kelembaban sangat tinggi. Penyebaran cepat melalui spora yang terbawa angin dan percikan air.",
        "solusi": [
            "Pilih varietas tomat yang tahan terhadap busuk daun.",
            "Gunakan bibit sehat dan bebas penyakit.",
            "Jaga jarak tanam agar sirkulasi udara baik dan daun cepat kering.",
            "Hindari penyiraman dari atas (membasahi daun), siram langsung ke tanah.",
            "Buang dan musnahkan bagian tanaman yang terinfeksi segera.",
            "Aplikasikan fungisida preventif (seperti produk berbahan aktif tembaga, chlorothalonil, mancozeb) sebelum gejala muncul, terutama jika cuaca mendukung. Fungisida sistemik mungkin diperlukan jika penyakit sudah menyebar."
        ]
    },
    "Tomato_Leaf_Mold": {
        "penyebab": "Disebabkan oleh jamur Passalora fulva (sebelumnya dikenal sebagai Fulvia fulva atau Cladosporium fulvum). Penyakit ini sangat menyukai kondisi kelembaban udara yang tinggi (di atas 85%) dan suhu sedang (22-24°C), sering terjadi di greenhouse atau area dengan sirkulasi udara buruk.",
        "solusi": [
            "Tanam varietas tomat yang resisten terhadap kapang daun.",
            "Tingkatkan sirkulasi udara dengan jarak tanam yang cukup dan pemangkasan daun bagian bawah.",
            "Turunkan kelembaban di greenhouse dengan ventilasi yang baik dan hindari penyiraman berlebih.",
            "Hindari membasahi daun saat menyiram.",
            "Buang daun terinfeksi untuk mengurangi sumber spora.",
            "Aplikasikan fungisida jika diperlukan, terutama pada tahap awal infeksi atau sebagai tindakan preventif di lingkungan berisiko tinggi."
        ]
    },
    "Tomato_Septoria_leaf_spot": {
        "penyebab": "Disebabkan oleh jamur Septoria lycopersici. Jamur ini bertahan pada sisa-sisa tanaman tomat atau gulma dari famili Solanaceae yang terinfeksi. Spora disebarkan oleh percikan air hujan, irigasi overhead, dan peralatan. Kelembaban tinggi dan suhu sedang (20-25°C) mendukung perkembangan penyakit.",
        "solusi": [
            "Rotasi tanaman minimal 3 tahun dengan tanaman non-Solanaceae.",
            "Sanitasi kebun secara menyeluruh, bersihkan dan musnahkan sisa tanaman setelah panen.",
            "Gunakan mulsa untuk mengurangi percikan tanah ke daun.",
            "Pastikan sirkulasi udara baik dengan jarak tanam dan pemangkasan.",
            "Hindari penyiraman dari atas.",
            "Aplikasikan fungisida (seperti yang mengandung chlorothalonil, mancozeb, atau tembaga) secara preventif atau saat gejala pertama muncul."
        ]
    },
    "Tomato_Spider_mites_Two_spotted_mite": {
        "penyebab": "Disebabkan oleh tungau laba-laba Tetranychus urticae. Hama ini sangat kecil dan berkembang biak dengan cepat pada kondisi panas dan kering. Mereka menghisap cairan sel daun, menyebabkan bintik-bintik kuning atau perunggu dan jaring halus.",
        "solusi": [
            "Jaga kelembaban di sekitar tanaman, karena tungau tidak suka kondisi lembab (namun hati-hati agar tidak memicu penyakit jamur).",
            "Semprot tanaman dengan air bertekanan untuk menjatuhkan tungau dari daun.",
            "Gunakan musuh alami seperti tungau predator (Phytoseiulus persimilis) atau kumbang predator.",
            "Aplikasikan sabun insektisida, minyak nimba, atau minyak hortikultura. Pastikan menyemprot bagian bawah daun tempat tungau berkumpul.",
            "Jika serangan parah, gunakan akarisida spesifik sesuai anjuran. Lakukan rotasi bahan aktif untuk mencegah resistensi."
        ]
    },
    "Tomato_Target_Spot": {
        "penyebab": "Disebabkan oleh jamur Corynespora cassiicola. Jamur ini dapat bertahan pada sisa-sisa tanaman dan menyebar melalui spora yang terbawa angin atau percikan air. Kelembaban tinggi dan suhu hangat hingga panas (20-30°C) mendukung infeksi dan perkembangan penyakit.",
        "solusi": [
            "Rotasi tanaman dengan tanaman yang bukan inang jamur ini.",
            "Praktikkan sanitasi kebun yang baik, termasuk menghilangkan sisa-sisa tanaman.",
            "Pastikan drainase yang baik dan hindari penyiraman berlebihan.",
            "Tingkatkan sirkulasi udara di sekitar tanaman.",
            "Aplikasikan fungisida yang efektif terhadap Corynespora cassiicola. Fungisida berbahan aktif mancozeb, chlorothalonil, atau strobilurin dapat dipertimbangkan, sesuai dengan rekomendasi setempat."
        ]
    },
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus": {
        "penyebab": "Disebabkan oleh Tomato Yellow Leaf Curl Virus (TYLCV) yang ditularkan oleh kutu kebul (Bemisia tabaci). Virus ini tidak ditularkan melalui benih. Tanaman terinfeksi menunjukkan gejala daun menguning, menggulung ke atas, kerdil, dan produksi buah menurun drastis.",
        "solusi": [
            "Tanam varietas tomat yang tahan atau toleran terhadap TYLCV.",
            "Kendalikan populasi kutu kebul, vektor utama virus. Gunakan insektisida yang efektif terhadap kutu kebul, perangkap kuning lengket, atau musuh alami.",
            "Gunakan mulsa plastik perak untuk menghalau kutu kebul.",
            "Sanitasi lingkungan, bersihkan gulma yang bisa menjadi inang kutu kebul.",
            "Cabut dan musnahkan tanaman yang terinfeksi segera untuk mengurangi sumber virus dan penyebaran oleh kutu kebul.",
            "Pada area endemik, pertimbangkan penggunaan screenhouse untuk melindungi tanaman dari kutu kebul."
        ]
    },
    "Tomato_Tomato_mosaic_virus": {
        "penyebab": "Disebabkan oleh Tomato Mosaic Virus (ToMV). Virus ini sangat mudah menular secara mekanis melalui sentuhan, peralatan pertanian, benih yang terinfeksi, dan kadang-kadang oleh pekerja. Virus dapat bertahan lama pada sisa-sisa tanaman kering dan tanah.",
        "solusi": [
            "Gunakan benih yang sehat dan bersertifikat bebas virus.",
            "Cuci tangan dengan sabun sebelum dan sesudah menangani tanaman, terutama jika berpindah dari satu area ke area lain.",
            "Sterilkan peralatan pertanian (pisau, gunting) secara berkala menggunakan disinfektan.",
            "Hindari penggunaan produk tembakau di sekitar tanaman tomat karena virus ini berkerabat dekat dengan Tobacco Mosaic Virus (TMV) dan dapat menular dari produk tembakau.",
            "Cabut dan musnahkan tanaman yang terinfeksi untuk mencegah penyebaran lebih lanjut.",
            "Rotasi tanaman tidak terlalu efektif karena virus dapat bertahan di tanah, namun tetap praktikkan sanitasi yang baik."
        ]
    },
    "Pepper_bell__healthy": {
        "penyebab": "Tanaman dalam kondisi sehat, tidak terinfeksi penyakit atau hama yang terdeteksi.",
        "solusi": [
            "Lanjutkan praktik budidaya yang baik untuk menjaga kesehatan tanaman.",
            "Lakukan pemantauan rutin terhadap hama dan penyakit.",
            "Pastikan tanaman mendapatkan nutrisi, air, dan cahaya matahari yang cukup.",
            "Jaga kebersihan area tanam."
        ]
    },
    "Potato_healthy": {
        "penyebab": "Tanaman dalam kondisi sehat, tidak terinfeksi penyakit atau hama yang terdeteksi.",
        "solusi": [
            "Pertahankan praktik agronomi yang baik.",
            "Lakukan monitoring secara berkala untuk deteksi dini masalah.",
            "Pastikan kebutuhan dasar tanaman terpenuhi (air, nutrisi, cahaya)."
        ]
    },
    "Tomato_healthy": {
        "penyebab": "Tanaman dalam kondisi sehat, tidak terinfeksi penyakit atau hama yang terdeteksi.",
        "solusi": [
            "Terus jaga kesehatan tanaman dengan pemupukan yang seimbang dan irigasi yang tepat.",
            "Lakukan inspeksi rutin untuk antisipasi hama atau penyakit.",
            "Optimalkan sirkulasi udara di sekitar tanaman."
        ]
    }
}

# --- Preprocessing Function ---
def preprocess_image(image: Image.Image) -> np.ndarray:
    if image.mode != "RGB":
        image = image.convert("RGB")
    target_size = (128, 128)
    image = image.resize(target_size)
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- CSS Kustom ---
light_theme_css_vars = """
    --primary-color: #2E8B57;
    --primary-hover-color: #256d42;
    --background-color: #f9f9f9;
    --secondary-background-color: #ffffff;
    --text-color: #333333;
    --secondary-text-color: #4A4A4A;
    --border-color: #e0e0e0;
    --info-box-bg: #f0f8ff;
    --info-box-border: #2E8B57;
    --button-text-color: white;
    --button-secondary-bg: #f0f2f6;
    --button-secondary-hover-bg: #e0e2e6;
    --button-secondary-text-color: #555;
    --box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    --box-shadow-light: 0 1px 3px rgba(0,0,0,0.05);
"""

dark_theme_css_vars = """
    --primary-color: #38a169;
    --primary-hover-color: #2f855a;
    --background-color: #1a202c;
    --secondary-background-color: #2d3748;
    --text-color: #e2e8f0;
    --secondary-text-color: #a0aec0;
    --border-color: #4a5568;
    --info-box-bg: #2c3e50;
    --info-box-border: #38a169;
    --button-text-color: white;
    --button-secondary-bg: #4a5568;
    --button-secondary-hover-bg: #2d3748;
    --button-secondary-text-color: #e2e8f0;
    --box-shadow: 0 2px 4px rgba(0,0,0,0.4);
    --box-shadow-light: 0 1px 3px rgba(0,0,0,0.2);
"""

if 'theme' not in st.session_state:
    st.session_state.theme = 'Light'

current_theme_vars = light_theme_css_vars if st.session_state.theme == 'Light' else dark_theme_css_vars

st.markdown(f"""
<style>
    :root {{
        {current_theme_vars}
    }}
    .stApp {{
        background-color: var(--background-color);
        color: var(--text-color);
    }}
    .stRadio {{
        display: flex;
        justify-content: center;
        margin-bottom: 2rem;
        padding: 0.5rem;
        background-color: var(--secondary-background-color);
        border-radius: 8px;
        box-shadow: var(--box-shadow-light);
    }}
    .stRadio > label {{ display: none; }}
    .stRadio > div[role="radiogroup"] {{
        display: flex;
        flex-direction: row !important;
        gap: 5px;
    }}
    .stRadio div[data-baseweb="radio"] label div:first-child {{
        padding: 0.5rem 1.5rem !important;
        border: 1px solid var(--border-color) !important;
        background-color: var(--secondary-background-color) !important;
        color: var(--secondary-text-color) !important;
        border-radius: 8px !important;
        transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease;
        box-shadow: none !important;
    }}
    .stRadio div[data-baseweb="radio"] label div:first-child:hover {{
        background-color: var(--primary-hover-color) !important;
        color: var(--button-text-color) !important;
        border-color: var(--primary-hover-color) !important;
    }}
    .stRadio div[data-baseweb="radio"] input:checked + div {{
        background-color: var(--primary-color) !important;
        color: var(--button-text-color) !important;
        border-color: var(--primary-color) !important;
        font-weight: bold;
    }}
    .stRadio div[data-baseweb="radio"] label span {{ display: none !important; }}
    .stButton>button {{
        background-color: var(--primary-color);
        color: var(--button-text-color);
        border-radius: 8px; padding: 10px 20px; border: none;
        box-shadow: var(--box-shadow);
        transition: background-color 0.3s ease;
        width: 100%;
    }}
    .stButton>button:hover {{
        background-color: var(--primary-hover-color);
        color: var(--button-text-color);
    }}
    .stButton>button:focus {{
        outline: none !important;
        box-shadow: 0 0 0 0.2rem var(--primary-color) !important;
    }}
    .stButton button[kind="secondary"] {{
        background-color: var(--button-secondary-bg);
        color: var(--button-secondary-text-color);
        border: 1px solid var(--border-color);
    }}
    .stButton button[kind="secondary"]:hover {{
        background-color: var(--button-secondary-hover-bg);
        color: var(--button-secondary-text-color);
    }}
    .title-app {{ color: var(--primary-color); text-align: center; font-weight: bold; margin-bottom: 0.5rem; }}
    .subtitle-app {{ text-align: center; color: var(--secondary-text-color); margin-bottom: 2rem; }}
    .expander-custom {{
        border: 1px solid var(--border-color); border-radius: 8px; padding: 15px;
        background-color: var(--secondary-background-color);
        box-shadow: var(--box-shadow-light); margin-bottom: 1rem;
    }}
    .stFileUploader label {{ font-weight: bold; color: var(--text-color); }}
    .stImage > img {{ border-radius: 8px; box-shadow: var(--box-shadow); }}
    .info-box {{
        background-color: var(--info-box-bg); border-left: 5px solid var(--info-box-border);
        padding: 10px 15px; margin-top: 15px; border-radius: 5px;
        box-shadow: var(--box-shadow-light);
    }}
    .info-box h4 {{ color: var(--primary-color); margin-top: 0; }}
    .info-box ul {{ padding-left: 20px; }}
    div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] div[data-testid="stExpander"] {{
        background-color: var(--secondary-background-color);
    }}
    h3 {{ color: var(--primary-color); }}
</style>
""", unsafe_allow_html=True)

# --- Navigasi Horizontal ---
col_nav1, col_nav2, col_nav3 = st.columns([0.15, 0.7, 0.15])

with col_nav1:
    st.image("https://emojigraph.org/media/apple/leafy-green_1f96c.png", width=50)

with col_nav2:
    page_selection = st.radio(
        "Navigasi:",
        ["🏡 Identifikasi Tanaman", "ℹ️ Tentang AgroDetect", "👥 Detail Proyek"],
        horizontal=True,
        label_visibility="collapsed"
    )

with col_nav3:
    if st.button(f"Mode {'Gelap' if st.session_state.theme == 'Light' else 'Terang'}", key="theme_button", use_container_width=True):
        st.session_state.theme = 'Dark' if st.session_state.theme == 'Light' else 'Light'
        st.rerun()

# --- Konten Halaman ---
if page_selection == "🏡 Identifikasi Tanaman":
    st.markdown("<h1 class='title-app'>🌿 AgroDetect</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='subtitle-app'>Aplikasi Identifikasi Hama dan Penyakit Daun Paprika, Tomat, dan Kentang</p>",
        unsafe_allow_html=True
    )
    st.header("🔍 Unggah Gambar untuk Identifikasi")
    uploaded_file = st.file_uploader(
        "Pilih gambar daun tanaman:", type=["jpg", "jpeg", "png"],
        help="Pastikan gambar jelas dan fokus pada daun yang terindikasi.",
        label_visibility="visible"
    )
    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar yang Diunggah", use_container_width=True)
        else:
            st.info("Silakan unggah gambar daun untuk memulai identifikasi.")
    with col2:
        if uploaded_file is not None:
            if 'identification_done' not in st.session_state:
                st.session_state.identification_done = False
            if 'predicted_class_name_state' not in st.session_state:
                st.session_state.predicted_class_name_state = None
            if 'confidence_state' not in st.session_state:
                st.session_state.confidence_state = None
            if 'predictions_state' not in st.session_state:
                st.session_state.predictions_state = None

            if st.button("🚀 Lakukan Identifikasi", key="identify_button", help="Klik untuk memulai proses identifikasi"):
                with st.spinner("⏳ Menganalisis gambar..."):
                    try:
                        processed_image = preprocess_image(image)
                        predictions = model.predict(processed_image)
                        predicted_class_index = np.argmax(predictions, axis=1)[0]
                        confidence = predictions[0][predicted_class_index] * 100
                        st.session_state.predictions_state = predictions
                        st.session_state.confidence_state = confidence
                        st.session_state.identification_done = True
                        if confidence >= CONFIDENCE_THRESHOLD:
                            predicted_class_name = CLASS_NAMES[predicted_class_index]
                            st.session_state.predicted_class_name_state = predicted_class_name
                        else:
                            st.session_state.predicted_class_name_state = "Tidak dapat mengidentifikasi penyakit/hama dengan pasti."
                    except AttributeError as ae:
                        st.error(f"Terjadi kesalahan: Model tidak berhasil dimuat. Detail: {ae}", icon="❌")
                        st.info("Pastikan file model 'best_model.keras' ada dan tidak rusak.")
                        st.session_state.identification_done = False
                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat memproses gambar atau memprediksi: {e}", icon="❌")
                        st.info("Pastikan gambar yang diunggah sesuai dan model Anda sudah dilatih dengan benar.")
                        st.session_state.identification_done = False
            
            if st.session_state.identification_done:
                st.subheader("📊 Hasil Identifikasi:")
                confidence = st.session_state.confidence_state
                predicted_class_name = st.session_state.predicted_class_name_state
                predictions = st.session_state.predictions_state

                if confidence >= CONFIDENCE_THRESHOLD and predicted_class_name != "Tidak dapat mengidentifikasi penyakit/hama dengan pasti.":
                    st.success(f"**Identifikasi:** {predicted_class_name.replace('_', ' ')}", icon="✅")
                    st.metric(label="Keyakinan Model", value=f"{confidence:.2f}%")
                    st.info(
                        "Klasifikasi ini didasarkan pada analisis fitur visual dari foto yang diunggah, "
                        "seperti pola bercak dan perubahan warna, untuk identifikasi dini yang akurat dan cepat.",
                        icon="💡"
                    )
                    if predicted_class_name in disease_info:
                        info = disease_info[predicted_class_name]
                        st.markdown("---")
                        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                        st.markdown(f"<h4>🌿 Informasi: {predicted_class_name.replace('_', ' ')}</h4>", unsafe_allow_html=True)
                        st.markdown(f"**Penyebab:** {info['penyebab']}")
                        st.markdown("**Solusi dan Penanganan:**")
                        for sol in info['solusi']:
                            st.markdown(f"- {sol}")
                        st.markdown("</div>", unsafe_allow_html=True)
                    elif "healthy" in predicted_class_name.lower():
                        info_sehat = disease_info.get(predicted_class_name)
                        if info_sehat:
                            st.markdown("---")
                            st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                            st.markdown(f"<h4>🌿 Informasi: {predicted_class_name.replace('_', ' ')} (Sehat)</h4>", unsafe_allow_html=True)
                            st.markdown(f"{info_sehat['penyebab']}")
                            st.markdown("**Rekomendasi:**")
                            for sol in info_sehat['solusi']:
                                st.markdown(f"- {sol}")
                            st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.warning(
                        f"**Identifikasi:** {predicted_class_name.replace('_', ' ')}", icon="⚠️"
                    )
                    if confidence is not None:
                        st.metric(label="Keyakinan Model (Tertinggi)", value=f"{confidence:.2f}%", delta=f"Di bawah {CONFIDENCE_THRESHOLD}%", delta_color="inverse")
                    st.info(
                        "Model tidak cukup yakin. Coba unggah gambar yang lebih jelas atau dari sudut yang berbeda. "
                        "Pastikan gambar fokus pada area daun yang menunjukkan gejala.",
                        icon="↪️"
                    )
                
                st.markdown("---")
                if predictions is not None:
                    with st.expander("Lihat Detail Probabilitas per Kelas", expanded=False):
                        sorted_indices = np.argsort(predictions[0])[::-1]
                        actual_predicted_class_index = np.argmax(predictions, axis=1)[0]
                        for i in sorted_indices:
                            prob = predictions[0][i] * 100
                            class_name_detail = CLASS_NAMES[i].replace('_', ' ')
                            if (i == actual_predicted_class_index and confidence >= CONFIDENCE_THRESHOLD and predicted_class_name != "Tidak dapat mengidentifikasi penyakit/hama dengan pasti."):
                                st.markdown(f"- **{class_name_detail}: {prob:.2f}% (Teridentifikasi)**")
                            elif (i == actual_predicted_class_index and confidence < CONFIDENCE_THRESHOLD):
                                st.markdown(f"- *{class_name_detail}: {prob:.2f}% (Keyakinan tertinggi)*")
                            else:
                                st.write(f"- {class_name_detail}: {prob:.2f}%")

            if st.button("🧹 Bersihkan", help="Klik untuk menghapus gambar dan hasil", type="secondary"):
                st.session_state.identification_done = False
                st.session_state.predicted_class_name_state = None
                st.session_state.confidence_state = None
                st.session_state.predictions_state = None
                st.rerun()

elif page_selection == "ℹ️ Tentang AgroDetect":
    st.markdown("<h1 class='title-app'>💡 Tentang AgroDetect</h1>", unsafe_allow_html=True)
    st.markdown( # PASTIKAN KONTEN INI LENGKAP
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
            <li>Menampilkan informasi penyebab dan solusi untuk penyakit yang terdeteksi.</li>
            <li>Antarmuka pengguna yang sederhana dan mudah digunakan.</li>
        </ul>
        <strong>Dataset yang Digunakan:</strong>
        Dataset yang digunakan untuk melatih model adalah <a href="https://www.kaggle.com/datasets/arjuntejaswi/plant-village" target="_blank" style="color: var(--primary-color);">Plant Village</a>.
        </div>
        """,
        unsafe_allow_html=True,
    )

elif page_selection == "👥 Detail Proyek":
    st.markdown("<h1 class='title-app'>👥 Detail Proyek Laskar AI</h1>", unsafe_allow_html=True)
    st.markdown( # PASTIKAN KONTEN INI LENGKAP
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
    <div style="text-align: center; color: var(--secondary-text-color); font-size: 0.9em;">
        © 2025 Tim Capstone Laskar AI (LAI25-RM097)
    </div>
    <br>
    """,
    unsafe_allow_html=True
)
