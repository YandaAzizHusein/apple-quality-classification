# **Laporan Proyek Machine Learning — Yanda Aziz Husein**

---

## **A. Domain Proyek**

Perkembangan sektor pertanian modern ditandai oleh meningkatnya volume produksi, kompleksitas rantai pasok, serta tuntutan pasar global terhadap mutu hasil pertanian yang konsisten. Dalam industri hortikultura, khususnya komoditas buah apel, kualitas menjadi faktor utama yang menentukan nilai jual, penerimaan pasar, dan daya saing produk.  

Namun, proses penilaian mutu yang hingga kini masih dilakukan secara manual melalui observasi visual cenderung bersifat subjektif, sulit distandarisasi, dan tidak efisien untuk diterapkan pada skala industri besar (Fadiji et al., 2023). Variasi antarpenilai sering menimbulkan ketidakkonsistenan dan meningkatkan risiko kesalahan klasifikasi mutu, yang pada akhirnya dapat memengaruhi kredibilitas produsen di mata konsumen.  

Seiring berkembangnya konsep *precision agriculture*, diperlukan inovasi berbasis teknologi data yang mampu menghasilkan evaluasi mutu secara objektif, cepat, dan berkelanjutan. Integrasi kecerdasan buatan (Artificial Intelligence / AI) dan *machine learning* telah terbukti meningkatkan akurasi penilaian hasil panen serta mengurangi ketergantungan terhadap faktor manusia (Grabska et al., 2023).  

Proyek ini mengimplementasikan algoritma *machine learning* berbasis data numerik dari **Apple Quality Dataset** (Nelgiriyewithana, n.d.) untuk mengklasifikasikan mutu buah apel secara otomatis berdasarkan atribut fisik dan sensorik seperti *Size*, *Weight*, *Sweetness*, *Crunchiness*, *Juiciness*, *Ripeness*, dan *Acidity*.  

Tiga model *supervised learning* diterapkan — **Logistic Regression**, **Random Forest**, dan **XGBoost** — dengan evaluasi menggunakan pembagian data *train–test* (80:20) serta metrik *accuracy*, *precision*, *recall*, dan *F1-score*.  

Melalui proyek ini, penilaian kualitas apel dapat dilakukan secara lebih cepat, akurat, dan konsisten tanpa intervensi subjektif, mendukung ekosistem *smart agriculture* yang lebih efisien, objektif, dan berkelanjutan.

**Referensi:**
- Fadiji, T., Bokaba, T., Fawole, O. A., & Twinomurinzi, H. (2023). *Artificial intelligence in postharvest agriculture: Mapping a global research agenda*. Frontiers in Sustainable Food Systems, 7, 1226583. https://doi.org/10.3389/fsufs.2023.1226583  
- Grabska, J., Niewiadomska, A., & Wójcik, M. (2023). *Analyzing the quality parameters of apples by spectroscopy from Vis/NIR to NIR region: A comprehensive review*. Foods, 12(9), 1770. https://doi.org/10.3390/foods12091770  
- Kavuncuoğlu, E., Yetkin, M., Özdemir, E., & Ozturk, M. (2023). *Exploration of machine learning algorithms for pH and moisture prediction in apples using VIS-NIR imaging*. Applied Sciences, 13(16), 8391. https://doi.org/10.3390/app13168391  
- Nelgiriyewithana, W. (n.d.). *Apple Quality Dataset*. Kaggle. https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality  

---

## **B. Business Understanding**

### **1. Problem Statements**
- Bagaimana membangun model *machine learning* yang mampu mengklasifikasikan kualitas apel (*good/bad*) secara akurat berdasarkan karakteristik hasil pengukuran fisik dan sensorik?
- Bagaimana memastikan model memiliki tingkat akurasi dan stabilitas yang memadai agar dapat diimplementasikan secara andal dalam industri penilaian mutu buah?

### **2. Goals**
- Mengembangkan model *machine learning* yang dapat memprediksi kualitas apel secara otomatis berdasarkan fitur numerik hasil pengukuran sensor.
- Menganalisis dan membandingkan performa beberapa algoritma klasifikasi guna menentukan model dengan hasil terbaik dan efisiensi tertinggi.

### **3. Solution Statements**
Proyek ini menggunakan dataset **Apple Quality** dari Kaggle dengan pendekatan *supervised learning* menggunakan tiga algoritma utama: Logistic Regression, Random Forest, dan XGBoost.  
Evaluasi performa model dilakukan menggunakan metrik *accuracy*, *precision*, *recall*, dan *F1-score* untuk menentukan algoritma terbaik yang mampu memberikan hasil prediksi paling akurat, efisien, dan konsisten dalam sistem evaluasi mutu apel otomatis.

---

## **C. Import Library**

| Kategori | Library | Fungsi Utama |
|-----------|----------|--------------|
| Manipulasi Data | `pandas`, `numpy` | Membaca, mengolah, dan memanipulasi dataset |
| Visualisasi | `matplotlib`, `seaborn` | Membuat grafik distribusi dan korelasi |
| Preprocessing | `SimpleImputer`, `StandardScaler`, `train_test_split` | Menangani nilai hilang, standarisasi, dan pembagian data |
| Modeling | `LogisticRegression`, `RandomForestClassifier`, `xgboost` | Membangun dan melatih model klasifikasi |
| Evaluasi | `accuracy_score`, `classification_report`, `confusion_matrix` | Mengukur performa model dengan berbagai metrik |

---

## **D. Data Understanding**

### **1. Deskripsi Dataset**
Dataset berasal dari Kaggle (*Apple Quality Dataset* oleh Nidula Elgiriyewithana). Dataset berisi 4.000 sampel dengan 9 kolom yang merepresentasikan atribut fisik dan sensorik apel seperti *Size*, *Weight*, *Sweetness*, *Crunchiness*, *Juiciness*, *Ripeness*, dan *Acidity*.  
Setiap sampel diklasifikasikan menjadi dua kategori: *good* dan *bad*.

### **2. Karakteristik Data**
- Jumlah data: **4.000 baris × 9 kolom**  
- Jenis fitur: 8 fitur independen, 1 label target  
- Distribusi target: seimbang (≈50% *good* dan *bad*)  
- Nilai kosong: tidak ditemukan (clean dataset)  
- Normalisasi: data telah *scaled* oleh penyedia dataset  

### **3. Eksplorasi Awal Data**
Dataset telah dinormalisasi (nilai mean mendekati 0 dan std ≈ 1).  
Tidak ditemukan anomali ekstrem, sehingga dataset siap untuk tahap *Data Preparation*.

---

## **E. Data Preparation**

### **1️⃣ Pembersihan Data**
Langkah-langkah:
- Menghapus kolom dengan korelasi rendah: `A_id`, `Weight`, `Crunchiness`, `Acidity`.  
- Menghapus kolom `Quality` karena sudah dikonversi ke `Quality_num`.  
- Menghapus baris terakhir yang berisi teks non-data.  
- Menghapus baris berisi *NaN*.

### **2️⃣ Pembagian Data**
Dataset dibagi menjadi **80% data latih (3.200 baris)** dan **20% data uji (800 baris)** menggunakan `train_test_split` dengan `stratify=y`.

### **3️⃣ Hasil Preparation**
Dataset akhir berisi 4 fitur relevan:
- `Size`
- `Sweetness`
- `Juiciness`
- `Ripeness`

Semua kolom bertipe numerik, tanpa nilai kosong, dan siap digunakan untuk tahap *Modeling*.

---

## **F. Modeling**

Tiga algoritma klasifikasi digunakan:
1. **Logistic Regression**  
   Model baseline yang sederhana dan cepat dengan interpretabilitas tinggi.

2. **Random Forest Classifier**  
   Model *ensemble learning* berbasis *decision trees* yang akurat dan stabil.

3. **XGBoost Classifier**  
   Model *gradient boosting* dengan kemampuan *regularization* untuk menghindari *overfitting*.

---

## **G. Evaluation**

### **1. Metrik Evaluasi**
- *Accuracy* — proporsi prediksi yang benar  
- *Precision* — ketepatan prediksi kelas positif  
- *Recall* — kemampuan menangkap seluruh data positif  
- *F1-score* — keseimbangan antara *precision* dan *recall*  
- *Confusion Matrix* — analisis kesalahan klasifikasi  

### **2. Hasil Evaluasi**

| Model | Accuracy | Precision | Recall | F1-score |
|:------|:---------:|:----------:|:--------:|:---------:|
| Logistic Regression | 0.7088 | 0.7098 | 0.7099 | 0.7094 |
| Random Forest | **0.7837** | **0.7871** | **0.7879** | **0.7875** |
| XGBoost | 0.7725 | 0.7770 | 0.7762 | 0.7761 |

### **3. Analisis**
- **Random Forest** memberikan performa terbaik dengan akurasi **78.37%**, stabil dan tidak *overfit*.  
- **XGBoost** mendekati performa Random Forest, dengan potensi peningkatan lewat *hyperparameter tuning*.  
- **Logistic Regression** berfungsi sebagai baseline dengan performa moderat.

---

## **H. Kesimpulan**

Berdasarkan hasil keseluruhan proyek **Apple Quality Classification**, diperoleh beberapa poin utama:

1. **Implementasi Machine Learning Berhasil Dilakukan**  
   Proyek berhasil mengaplikasikan algoritma *machine learning* untuk mengklasifikasikan mutu apel secara otomatis, objektif, dan efisien.

2. **Model dengan Performa Terbaik**  
   Model **Random Forest** memberikan hasil terbaik dengan akurasi **78.37%**, disusul oleh **XGBoost** dan **Logistic Regression**.

3. **Fitur Paling Berpengaruh**  
   Fitur **Size**, **Sweetness**, **Juiciness**, dan **Ripeness** memiliki pengaruh dominan terhadap kualitas apel, sementara **Weight**, **Crunchiness**, dan **Acidity** memiliki kontribusi rendah.

4. **Relevansi terhadap Industri Pertanian Modern**  
   Penerapan *machine learning* terbukti mampu meningkatkan efisiensi dan mengurangi subjektivitas dalam proses penilaian kualitas hasil pertanian.

5. **Potensi Pengembangan ke Depan**  
   Model dapat dioptimalkan melalui *hyperparameter tuning* dan pengembangan fitur berbasis citra atau sensor non-destruktif agar lebih adaptif terhadap kondisi nyata di industri pertanian modern.
