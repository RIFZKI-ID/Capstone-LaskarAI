# AgroDetect: Aplikasi Identifikasi Hama dan Penyakit Tanaman

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Selesai](https://img.shields.io/badge/status-100%25%20selesai-brightgreen)](#)
[![Framework: Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)](#)

[cite_start]AgroDetect adalah aplikasi berbasis web yang dirancang sebagai alat bantu digital bagi petani untuk mendeteksi hama dan penyakit pada daun tanaman paprika, tomat, dan kentang secara otomatis menggunakan *Machine Learning*[cite: 4, 14].

[cite_start]Proyek ini lahir untuk mengatasi kesulitan yang dihadapi petani dalam mengidentifikasi penyakit tanaman sejak dini, terutama bagi mereka yang tidak memiliki akses ke ahli pertanian[cite: 15]. [cite_start]Mengingat 20-40% produksi pangan global hilang akibat hama dan penyakit, deteksi dini menjadi sangat krusial[cite: 16]. [cite_start]AgroDetect memberikan solusi dengan diagnosis cepat dan akurat hanya dengan mengunggah foto daun tanaman[cite: 17, 19].

[cite_start]**[➡️ Kunjungi Aplikasi Live di Sini](https://capstone-laskarai.streamlit.app/)** [cite: 91]

![Tangkapan Layar AgroDetect](https://i.ibb.co/Sn2v0D2/Screenshot-2024-08-27-143329.png) 

## 📜 Daftar Isi
- [Fitur Utama](#-fitur-utama)
- [Latar Belakang](#-latar-belakang)
- [Teknologi yang Digunakan](#-teknologi-yang-digunakan)
- [Arsitektur Model](#-arsitektur-model)
- [Instalasi & Cara Menjalankan](#-instalasi--cara-menjalankan)
- [Dataset](#-dataset)
- [Tangkapan Layar Aplikasi](#-tangkapan-layar-aplikasi)
- [Tim Pengembang](#-tim-pengembang)
- [Informasi Proyek](#-informasi-proyek)
- [Tautan Penting](#-tautan-penting)

---

## ✨ Fitur Utama

* [cite_start]**Deteksi Cepat & Akurat**: Menggunakan model *Convolutional Neural Network* (CNN) untuk memproses foto daun dan memberikan hasil klasifikasi penyakit secara cepat dan akurat[cite: 17].
* [cite_start]**Fokus pada Komoditas Penting**: Didesain khusus untuk mendeteksi penyakit pada tiga jenis tanaman: paprika, tomat, dan kentang[cite: 14, 128].
* [cite_start]**Rekomendasi Penanganan Awal**: Tidak hanya mendiagnosis, aplikasi ini juga menyediakan informasi lengkap mengenai penyebab, gejala, dan solusi penanganan awal yang praktis[cite: 18, 132].
* [cite_start]**Sistem Validasi Cerdas**: Hanya menampilkan hasil jika tingkat kepercayaan model tinggi (di atas 80%) untuk meningkatkan keandalan diagnosis dan menghindari hasil yang tidak pasti[cite: 130].
* [cite_start]**Antarmuka Sederhana**: Dibangun dengan Streamlit, antarmuka aplikasi sangat intuitif dan mudah digunakan, bahkan bagi pengguna tanpa latar belakang teknis[cite: 142].

---

## 🌾 Latar Belakang
[cite_start]Banyak petani menghadapi kesulitan dalam mengidentifikasi hama dan penyakit tanaman pada tahap awal, yang sering kali berujung pada kegagalan panen[cite: 15]. [cite_start]Menurut data FAO, sekitar 20-40% dari total produksi pangan global hilang karena masalah ini setiap tahunnya[cite: 16, 111]. [cite_start]Proses identifikasi manual tidak hanya lambat tetapi juga memerlukan keahlian khusus yang tidak selalu tersedia di lapangan[cite: 16]. [cite_start]AgroDetect hadir sebagai solusi cerdas untuk menjembatani kesenjangan ini, memberdayakan petani dengan teknologi AI untuk diagnosis instan dan praktis[cite: 14, 20].

---

## 🛠️ Teknologi yang Digunakan

* [cite_start]**Framework Aplikasi**: Streamlit [cite: 142]
* [cite_start]**Model Machine Learning**: Convolutional Neural Network (CNN) dengan arsitektur **MobileNetV2** [cite: 17, 143]
* **Pustaka Utama**: TensorFlow, Keras, NumPy, PIL
* **Deployment**: Streamlit Cloud

---

## 🧠 Arsitektur Model

[cite_start]AgroDetect menggunakan model *Deep Learning* dengan arsitektur **MobileNetV2** yang dikenal efisien dan canggih untuk tugas pengenalan gambar[cite: 143]. [cite_start]Model ini dilatih menggunakan dataset *Plant Village* yang masif dan beragam, berisi gambar daun tanaman sehat dan berpenyakit[cite: 60, 113]. [cite_start]Pada tahap validasi, model ini berhasil mencapai **tingkat akurasi sekitar 94%**[cite: 137].

---

## 🚀 Instalasi & Cara Menjalankan

Untuk menjalankan proyek ini di lingkungan lokal Anda, ikuti langkah-langkah berikut:

1.  **Clone repository ini:**
    ```bash
    git clone [https://github.com/RIFZKI-ID/Capstone-LaskarAl.git](https://github.com/RIFZKI-ID/Capstone-LaskarAl.git)
    cd Capstone-LaskarAl
    ```
   

2.  **Buat dan aktifkan virtual environment (disarankan):**
    ```bash
    python -m venv venv
    # Untuk Windows
    venv\Scripts\activate
    # Untuk macOS/Linux
    source venv/bin/activate
    ```

3.  **Instal semua dependensi yang dibutuhkan:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Catatan: Pastikan Anda sudah membuat file `requirements.txt` di repository Anda)*

4.  **Jalankan aplikasi Streamlit:**
    ```bash
    streamlit run app.py
    ```
    *(Asumsi nama file utama adalah `app.py`)*

---

## 📊 Dataset
[cite_start]Proyek ini menggunakan dataset **Plant Village** yang tersedia di Kaggle[cite: 89, 113]. [cite_start]Dataset ini merupakan sumber daya yang masif dan beragam, berisi gambar daun dari berbagai tanaman yang digunakan untuk melatih model CNN kami[cite: 60].

[cite_start]**[Lihat Dataset di Kaggle](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)** [cite: 89]

---

## 🖥️ Tangkapan Layar Aplikasi

<p align="center">
  <img src="https://i.ibb.co/mHk8LMD/image.png" width="80%">
  <br>
  <em>Tampilan Utama</em>
</p>
<p align="center">
  <img src="https://i.ibb.co/9vGjbF3/image.png" width="80%">
  <br>
  <em>Halaman Tentang AgroDetect</em>
</p>
<p align="center">
  <img src="https://i.ibb.co/cQhF5pP/image.png" width="80%">
  <br>
  <em>Halaman Tim Pengembang</em>
</p>


---

## 👥 Tim Pengembang
[cite_start]Kami adalah sekelompok individu yang bersemangat dalam menerapkan AI untuk solusi nyata[cite: 81]. Proyek ini adalah hasil kolaborasi dari:

| ID Peserta | Nama Lengkap | Universitas |
| :--- | :--- | :--- |
| `A180YBF358` | Mukhamad Ikhsanudin | [cite_start]Universitas Airlangga [cite: 6] |
| `A706YBF391` | Patuh Rujhan Al Istizhar | [cite_start]Universitas Swadaya Gunung Jati [cite: 7] |
| `A573YBF408` | Rahmat Hidayat | [cite_start]Universitas Lancang Kuning [cite: 8] |
| `A314YBF428` | Rifzki Adiyaksa | [cite_start]Universitas Singaperbangsa Karawang [cite: 9] |

---

## ℹ️ Informasi Proyek

* [cite_start]**ID Grup**: `LAI25-RM097` [cite: 5]
* [cite_start]**Tema Proyek**: Solusi Cerdas untuk Masa Depan yang Lebih Baik [cite: 5]
* [cite_start]**Advisor/Pembimbing**: Stevani Dwi Utomo [cite: 5]
* [cite_start]**Program**: Proyek ini dikembangkan sebagai Capstone untuk program Laskar AI [cite: 71][cite_start], didukung oleh AI Merdeka Lintasarta [cite: 2][cite_start], NVIDIA [cite: 3][cite_start], dan Dicoding[cite: 3].

---

## 🔗 Tautan Penting

* [cite_start]**[Link Repositori GitHub](https://github.com/RIFZKI-ID/Capstone-LaskarAl)** [cite: 93]
* [cite_start]**[Link Deployment Aplikasi](https://capstone-laskarai.streamlit.app/)** [cite: 91]
* [cite_start]**[Link Video Demo Proyek](https://link.rifzki.my.id/t/Capstone-Demo)** [cite: 24]
* [cite_start]**[Link Video Presentasi](https://drive.google.com/file/d/1Qd8FbZmjXiPSjIZMEsVsopLq1bWbZe9-/view?usp=sharing)** [cite: 96]
