# **Submission 2: Credit Card Customer Churn Prediction Pipeline & Cloud Monitoring**

Nama: Danuardi Saputro  
Username dicoding: dnnuuyzzo  

| | **Deskripsi** |
| ----------- | ----------- |
| **Dataset** | Dataset yang digunakan adalah [Credit Card Customers (Bank Churners)](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers) dari Kaggle oleh Sakshi Goyal.<br><br>**1. Ringkasan & Ruang Lingkup Dataset:**<br>Dataset ini memuat profil demografis, kondisi finansial perbankan, dan riwayat transaksi nasabah pemegang kartu kredit. Dataset dimanfaatkan untuk mengidentifikasi nasabah yang berpotensi menutup akun/kartu kredit (*attrited customer*) sehingga bank dapat menerapkan strategi retensi sedini mungkin.<br><br>**2. Jumlah Data & Fitur:**<br>- Jumlah Observasi: 1.000 baris data nasabah terverifikasi untuk proses pelatihan dan evaluasi pipeline.<br>- Jumlah Total Fitur: 20 fitur (1 fitur target label, 14 fitur numerik, dan 5 fitur kategorikal). Kolom identifier unik (`CLIENTNUM`) serta kolom benchmark Naive Bayes diabaikan.<br>- Kondisi Missing Value: 0 missing value (dataset lengkap).<br><br>**3. Rincian Fitur Berdasarkan Tipe Data:**<br>- Label / Target (1 Fitur): `Attrition_Flag` (1 = Attrited Customer / Churn, 0 = Existing Customer / Bertahan).<br>- Fitur Numerik (14 Fitur): `Customer_Age`, `Dependent_count`, `Months_on_book`, `Total_Relationship_Count`, `Months_Inactive_12_mon`, `Contacts_Count_12_mon`, `Credit_Limit`, `Total_Revolving_Bal`, `Avg_Open_To_Buy`, `Total_Amt_Chng_Q4_Q1`, `Total_Trans_Amt`, `Total_Trans_Ct`, `Total_Ct_Chng_Q4_Q1`, `Avg_Utilization_Ratio`.<br>- Fitur Kategorikal (5 Fitur): `Gender`, `Education_Level`, `Marital_Status`, `Income_Category`, `Card_Category`. |
| **Masalah** | Penurunan jumlah nasabah kartu kredit (*customer churn*) berdampak signifikan pada penurunan pendapatan transaksi perbankan dan meningkatkan biaya akuisisi nasabah baru (*Customer Acquisition Cost*). Pihak bank membutuhkan sistem otomatis yang andal dan berbasis cloud untuk memprediksi nasabah berisiko churn secara real-time, serta sistem monitoring metrik untuk memantau performa dan ketersediaan layanan inferensi model secara berkala. |
| **Solusi machine learning** | Membangun sistem *end-to-end* machine learning pipeline berbasis TensorFlow Extended (TFX) yang memvalidasi kualitas data secara otomatis, melatih model Deep Neural Network (DNN) dengan optimasi hyperparameter melalui KerasTuner, mengevaluasi performa dan keadilan (*fairness*) model menggunakan TFMA, mendistribusikan model ke TensorFlow Serving di platform cloud (Railway/Heroku/Render) menggunakan Docker, serta memantau kesehatan server serving menggunakan Prometheus dan Grafana. |
| **Metode pengolahan** | Menggunakan modul `modules/dnnuuyzzo_transform.py` dan komponen TFX Transform:<br>**1. Standarisasi Fitur Numerik:** Seluruh 14 fitur numerik diskalakan menggunakan penskalaan z-score (`tft.scale_to_z_score`) sehingga berpusat pada nilai mean 0 dan standar deviasi 1.<br>**2. Transformasi Fitur Kategorikal:** Seluruh 5 fitur kategorikal dikonversi menjadi representasi indeks integer (`tft.compute_and_apply_vocabulary`) dengan alokasi out-of-vocabulary (OOV) bucket.<br>**3. Transformasi Label Target:** Mengonversi label `Attrition_Flag` ke representasi `tf.int64` (1 untuk Churn, 0 untuk Existing).<br>**4. Pencegahan Training-Serving Skew:** Seluruh logika transformasi dikompilasi ke dalam `tf.Graph` hermetis yang disematkan langsung pada *serving signature* model (`serving_default`). |
| **Arsitektur model** | Arsitektur Deep Neural Network (DNN) berbasis `tf.keras` dengan konfigurasi optimal hasil tuning KerasTuner (`dnnuuyzzo-pipeline/Tuner/best_hyperparameters/11/best_hyperparameters.txt`):<br>**1. Input Layer:** Menerima 14 input numerik (`tf.float32`) dan 5 input kategorikal indeks (`tf.int64`).<br>**2. Categorical Embedding Layers:**<br>- `Gender`: dimensi embedding 4 unit.<br>- `Education_Level`: dimensi embedding 4 unit.<br>- `Marital_Status`: dimensi embedding 16 unit.<br>- `Income_Category`: dimensi embedding 16 unit.<br>- `Card_Category`: dimensi embedding 4 unit.<br>**3. Concatenation Layer:** Menggabungkan 14 fitur numerik ternormalisasi dan seluruh flattened embedding categorical menjadi satu representasi terpadu.<br>**4. Dense Hidden Layers:** 2 lapisan *Fully Connected (Dense)* dengan aktivasi ReLU dan Dropout:<br>- Hidden Layer 1 (`units_0`): 128 unit, Dropout (`dropout_0`): 0.2.<br>- Hidden Layer 2 (`units_1`): 32 unit, Dropout (`dropout_1`): 0.4.<br>**5. Output Layer:** 1 unit Dense dengan fungsi aktivasi Sigmoid untuk mengestimasi probabilitas churn nasabah.<br>**6. Optimizer & Loss:** Adam optimizer dengan *learning rate* 0.01 dan fungsi loss *BinaryCrossentropy*.<br>**7. Serving Signature:** Menyematkan hermetic `tft` preprocessing graph langsung ke signature `serving_default` model untuk menerima raw data via `tf.train.Example`. |
| **Metrik evaluasi** | Model dievaluasi menggunakan TensorFlow Model Analysis (TFMA) dengan metrik:<br>**1. AUC (Area Under ROC Curve):** Metrik utama klasifikasi biner dengan ambang batas minimal (*lower bound threshold*) >= 0.50.<br>**2. Binary Accuracy, Precision, dan Recall:** Mengukur akurasi prediksi, ketepatan deteksi churn, dan sensitivitas terhadap nasabah berisiko.<br>**3. Fairness Slicing:** Evaluasi slicing pada fitur `Gender` untuk memastikan performa model adil dan tidak bias terhadap kelompok gender tertentu.<br>**4. Validation Threshold:** Validasi kelayakan model terhadap baseline untuk penetapan status *Model Blessing*. |
| **Performa model** | Berdasarkan hasil evaluasi komprehensif menggunakan **TensorFlow Model Analysis (TFMA)** pada dataset evaluasi (331 data nasabah):<br>**1. Metrik Keseluruhan (Overall Evaluation):**<br>- **AUC (Area Under ROC Curve):** `1.0000` (Ambang batas minimum >= 0.50 tercapai sempurna).<br>- **Binary Accuracy:** `1.0000` (100.0%).<br>- **Precision:** `1.0000` (100.0%).<br>- **Recall:** `1.0000` (100.0%).<br>- **Loss:** `0.0105`.<br>**2. Evaluasi Keadilan Gender (Fairness Slicing):**<br>- `Gender = 'F'` (118 nasabah wanita): AUC = `1.0000`, Binary Accuracy = `1.0000` (100.0%), Precision = `1.0000`, Recall = `1.0000`, Loss = `0.0245`.<br>- `Gender = 'M'` (213 nasabah pria): AUC = `1.0000`, Binary Accuracy = `1.0000` (100.0%), Precision = `1.0000`, Recall = `1.0000`, Loss = `0.0027`.<br>**3. Status Model Blessing:** Model berstatus **`BLESSED`** (lolos seluruh validasi TFMA terhadap threshold dan baseline) dan otomatis diekspor oleh komponen Pusher ke direktori `serving_model_dir/credit_card_churn_model` untuk siap dideploy ke cloud. |
| **Opsi deployment** | Model dideploy menggunakan container TensorFlow Serving 2.11.0 melalui berkas `Dockerfile`. Pilihan deployment mencakup:<br>**1. Platform Cloud:** Container dideploy ke layanan cloud seperti Railway, Heroku, atau Render dengan port 8501 (REST API) dan 8500 (gRPC).<br>**2. Deployment Lokal:** Container dapat dijalankan secara lokal menggunakan Docker Engine (`docker run -p 8501:8501`). |
| **Web app** | Tautan web app model serving di cloud:<br>[credit-card-churn-serving-metadata](https://credit-card-churn-mlops-production.up.railway.app/v1/models/credit_card_churn_model/metadata)<br>*(Endpoint REST API Prediksi: `https://credit-card-churn-mlops-production.up.railway.app/v1/models/credit_card_churn_model:predict`)* |
| **Monitoring** | Monitoring performa sistem machine learning dilakukan menggunakan Prometheus dan Grafana:<br>**1. Prometheus:** Mengambil (*scraping*) metrik dari TensorFlow Serving (`/monitoring/prometheus/metrics`) secara berkala.<br>**2. Metrik yang Dipantau:** Latensi inferensi prediksi (`:predict`), total throughput request per detik, tingkat error code HTTP, dan utilisasi resource (CPU/Memory).<br>**3. Grafana Dashboard:** Menampilkan visualisasi interaktif dari metrik Prometheus untuk memantau performa dan ketersediaan sistem serving secara real-time. |

---

## Struktur Berkas Proyek

Berikut merupakan struktur berkas proyek yang disiapkan untuk pengiriman submission:

```text
dnnuuyzzo-submission/
├── dnnuuyzzo-pipeline/                  # Direktori seluruh artefak komponen TFX ML Pipeline
│   ├── CsvExampleGen/
│   ├── StatisticsGen/
│   ├── SchemaGen/
│   ├── ExampleValidator/
│   ├── Transform/
│   ├── Tuner/
│   ├── Trainer/
│   ├── Evaluator/
│   ├── Pusher/
│   └── metadata.sqlite
├── serving_model_dir/                   # Direktori model serving hasil ekspor Pusher
│   └── credit_card_churn_model/
├── modules/                             # Direktori seluruh modul pipeline (clean code)
│   ├── __init__.py
│   ├── dnnuuyzzo_transform.py           # Modul transformasi fitur
│   ├── dnnuuyzzo_trainer.py             # Modul pelatihan DNN dan serving signature
│   └── dnnuuyzzo_tuner.py               # Modul hyperparameter tuning via KerasTuner
├── monitoring/                          # Direktori konfigurasi monitoring Prometheus
│   ├── Dockerfile                       # Dockerfile Prometheus
│   ├── prometheus.yml                   # Konfigurasi target scraping Prometheus
│   ├── prometheus.config                # Berkas konfigurasi Prometheus
│   └── dnnuuyzzo-monitoring.png         # Bukti screenshot dashboard Prometheus
├── data/
│   └── credit_card_churn.csv            # Dataset 1.000 baris credit card churn
├── dnnuuyzzo_pipeline.ipynb             # Notebook eksekusi TFX Pipeline (sudah dijalankan)
├── dnnuuyzzo-testing.ipynb              # Notebook pengujian REST API inferensi di cloud
├── dnnuuyzzo-deployment.png             # Bukti screenshot model serving aktif di cloud
├── dnnuuyzzo-pylint.png                 # Bukti screenshot hasil penilaian Pylint pada modules
├── dnnuuyzzo-grafana-dashboard.png      # Bukti screenshot visualisasi dashboard Grafana
├── Dockerfile                           # Dockerfile TensorFlow Serving untuk cloud
├── requirements.txt                     # Daftar dependensi library proyek
└── README.md                            # Dokumentasi resmi proyek
```

---

## Panduan Menjalankan Sistem Machine Learning & Monitoring

### 1. Menjalankan Machine Learning Pipeline (TFX)
1. Siapkan environment Python sesuai dengan dependensi pada `requirements.txt`.
2. Buka dan jalankan alur notebook `dnnuuyzzo_pipeline.ipynb`.
3. Seluruh artefak komponen (ExampleGen hingga Pusher) akan dihasilkan secara otomatis ke dalam folder `dnnuuyzzo-pipeline/` dan model yang lolos evaluasi diekspor ke `serving_model_dir/credit_card_churn_model/`.

### 2. Deployment Model ke Cloud Menggunakan Docker
- **Build Docker Image:**
  ```bash
  docker build -t dnnuuyzzo-churn-serving:latest .
  ```
- **Jalankan Container Secara Lokal:**
  ```bash
  docker run -d -p 8501:8501 -p 8500:8500 --name tf_serving_churn dnnuuyzzo-churn-serving:latest
  ```
- **Deploy ke Cloud (Railway / Heroku / Render):**
  - Hubungkan repository GitHub ke layanan cloud pilihan Anda.
  - Platform cloud akan otomatis mendeteksi `Dockerfile` di root direktori dan melakukan build serta menjalankan container model serving.

### 3. Menguji Inferensi Model Serving via REST API
1. Buka dan jalankan notebook `dnnuuyzzo-testing.ipynb`.
2. Masukkan URL endpoint cloud atau lokal pada variabel `SERVER_BASE_URL`.
3. Notebook akan melakukan pengujian koneksi (*health check*), serialisasi data uji ke format `tf.train.Example` base64, mengirimkan request prediksi via HTTP POST, dan menampilkan estimasi probabilitas churn nasabah.

### 4. Menjalankan Monitoring dengan Prometheus & Grafana
- **Menjalankan Container Prometheus:**
  ```bash
  cd monitoring
  docker build -t dnnuuyzzo-prometheus:latest .
  docker run -d -p 9090:9090 --name prometheus_churn dnnuuyzzo-prometheus:latest
  ```
- **Akses Prometheus UI:**
  Buka `http://localhost:9090` pada web browser untuk memantau status target scraping dan metrik `/monitoring/prometheus/metrics`.
- **Integrasi Grafana Dashboard:**
  1. Jalankan container Grafana (`docker run -d -p 3000:3000 --name grafana grafana/grafana`).
  2. Tambahkan Prometheus sebagai data source (`http://localhost:9090`).
  3. Buat dashboard untuk memvisualisasikan `tensorflow_serving_request_latency_count` dan `process_cpu_seconds_total`.
