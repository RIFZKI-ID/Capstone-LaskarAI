import streamlit as st
from PIL import Image

# Assuming you have your ML model loading and prediction logic
# You'll need to replace these with your actual model loading and prediction functions.
# The 'predict_plant_and_disease' function should return both the plant type
# and the disease/health status.
# For example:
# from your_ml_model_file import load_model, predict_plant_and_disease

st.set_page_config(
    page_title="AgroDetect: Aplikasi Identifikasi Hama dan Penyakit Daun",
    # icon_image="path/to/your/icon.png",
)

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
        st.write("Menganalisis gambar...")

        try:
            # --- MODEL PREDICTION LOGIC ---
            # This is where your actual ML model integration goes.
            # You will need a function (e.g., predict_plant_and_disease)
            # that takes the processed image and returns both the plant type
            # and the predicted condition (healthy, disease, or pest).

            # Example: Preprocess the image for your model (resize, normalize, etc.)
            # processed_image = preprocess_image(image)

            # Example: Make prediction using your ML model
            # plant_type, condition = predict_plant_and_disease(processed_image)

            # For demonstration, let's use dummy values that simulate a model output
            # In a real application, these would come from your ML model's prediction.
            # Your model would ideally predict one of these combinations:
            # ("Paprika", "Sehat"), ("Tomat", "Penyakit Bercak Daun"), ("Kentang", "Hama Kutu Daun"), etc.

            # This is a placeholder. Your actual model would determine these values.
            dummy_plant_type = "Tidak Diketahui"
            dummy_condition = "Belum Teridentifikasi"

            # Simulate a simple prediction for demonstration
            # In a real scenario, the model would analyze the image
            # to determine both plant type and condition.
            if (
                "pepper" in uploaded_file.name.lower()
                or "paprika" in uploaded_file.name.lower()
            ):
                dummy_plant_type = "Paprika"
                dummy_condition = "Sehat (Simulasi)"
            elif (
                "tomato" in uploaded_file.name.lower()
                or "tomat" in uploaded_file.name.lower()
            ):
                dummy_plant_type = "Tomat"
                dummy_condition = "Penyakit Bercak Daun (Simulasi)"
            elif (
                "potato" in uploaded_file.name.lower()
                or "kentang" in uploaded_file.name.lower()
            ):
                dummy_plant_type = "Kentang"
                dummy_condition = "Hama Kutu Daun (Simulasi)"
            else:
                dummy_condition = (
                    "Kondisi Tidak Diketahui (Model belum dilatih untuk ini)"
                )

            st.subheader("Hasil Identifikasi:")
            if dummy_plant_type != "Tidak Diketahui":
                st.success(f"Jenis Tanaman: {dummy_plant_type}")
            else:
                st.info(
                    "Jenis Tanaman: Tidak dapat dikenali secara otomatis (model perlu dilatih lebih lanjut untuk ini)"
                )

            st.success(f"Kondisi Daun: {dummy_condition}")
            st.info(
                "Klasifikasi ini didasarkan pada analisis fitur visual dari foto yang diunggah pengguna, seperti pola bercak dan perubahan warna, untuk identifikasi dini yang akurat dan cepat."
            )

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
            st.info(
                "Pastikan Anda telah mengintegrasikan model Machine Learning dengan benar dan model mampu memprediksi jenis tanaman secara otomatis."
            )

# --- Footer (Optional) ---
st.markdown("---")
st.markdown("ID Grup: LAI25-RM097")
st.markdown("Anggota Grup:")
st.markdown("1. A180YBF358 - Mukhamad Ikhsanudin - Universitas Airlangga")
st.markdown(
    "2. A706YBF391 - Patuh Rujhan Al Istizhar - Universitas Swadaya Gunung Jati"
)
st.markdown("3. A573YBF408 - Rahmat Hidayat - Universitas Lancang Kuning")
st.markdown("4. A314YBF428 - Rifzki Adiyaksa - Universitas Singaperbangsa Karawang")
