# Apple Quality Classification - Yanda Aziz Husein
---
## **A. Domain Proyek**
Perkembangan sektor pertanian modern ditandai oleh meningkatnya volume produksi, kompleksitas rantai pasok, serta tuntutan pasar global terhadap mutu hasil pertanian yang konsisten. Dalam industri hortikultura, khususnya komoditas buah apel, kualitas menjadi faktor utama yang menentukan nilai jual, penerimaan pasar, dan daya saing produk. Namun, proses penilaian mutu yang hingga kini masih dilakukan secara manual melalui observasi visual cenderung bersifat subjektif, sulit distandarisasi, dan tidak efisien untuk diterapkan pada skala industri besar (Fadiji et al., 2023). Variasi antarpenilai sering menimbulkan ketidakkonsistenan dan meningkatkan risiko kesalahan klasifikasi mutu, yang pada akhirnya dapat memengaruhi kredibilitas produsen di mata konsumen.

Seiring berkembangnya konsep precision agriculture, diperlukan inovasi berbasis teknologi data yang mampu menghasilkan evaluasi mutu secara objektif, cepat, dan berkelanjutan. Integrasi kecerdasan buatan (Artificial Intelligence / AI) dan machine learning telah terbukti meningkatkan akurasi penilaian hasil panen serta mengurangi ketergantungan terhadap faktor manusia (Grabska et al., 2023). Ketidakefisienan metode manual tidak hanya memperlambat proses produksi, tetapi juga berdampak langsung pada rantai pasok — kesalahan klasifikasi dapat mencampur produk premium dengan non-premium, menurunkan reputasi produsen, serta menimbulkan kerugian ekonomi. Oleh karena itu, diperlukan sistem otomatis yang adaptif terhadap variasi lingkungan dan varietas buah untuk mendukung praktik pertanian cerdas yang efisien, akurat, dan berkelanjutan (Kavuncuoğlu et al., 2023).

Proyek ini mengimplementasikan algoritma machine learning berbasis data numerik dari Apple Quality Dataset (Nelgiriyewithana, n.d.) untuk mengklasifikasikan mutu buah apel secara otomatis berdasarkan atribut fisik dan sensorik seperti Size, Weight, Sweetness, Crunchiness, Juiciness, Ripeness, dan Acidity. Tiga model supervised learning diterapkan — Logistic Regression, Random Forest, dan XGBoost — dengan evaluasi menggunakan pembagian data train–test (80:20) serta metrik accuracy, precision, recall, dan F1-score. Pendekatan ini didukung oleh berbagai studi yang menunjukkan efektivitas algoritma berbasis decision tree ensembles dalam memprediksi mutu buah secara non-destruktif (Grabska et al., 2023; Kavuncuoğlu et al., 2023).

Melalui proyek ini, penilaian kualitas apel dapat dilakukan secara lebih cepat, akurat, dan konsisten tanpa intervensi subjektif. Selain meningkatkan efisiensi proses penentuan mutu, hasil penelitian ini diharapkan berkontribusi terhadap transformasi menuju ekosistem data-driven smart agriculture yang lebih objektif, inovatif, dan berkelanjutan.

### Referensi

Fadiji, T., Bokaba, T., Fawole, O. A., & Twinomurinzi, H. (2023). *Artificial intelligence in postharvest agriculture: Mapping a global research agenda*. Frontiers in Sustainable Food Systems, 7, 1226583. https://doi.org/10.3389/fsufs.2023.1226583  

Grabska, J., Niewiadomska, A., & Wójcik, M. (2023). *Analyzing the quality parameters of apples by spectroscopy from Vis/NIR to NIR region: A comprehensive review*. Foods, 12(9), 1770. https://doi.org/10.3390/foods12091770  

Kavuncuoğlu, E., Yetkin, M., Özdemir, E., & Ozturk, M. (2023). *Exploration of machine learning algorithms for pH and moisture prediction in apples using VIS-NIR imaging*. Applied Sciences, 13(16), 8391. https://doi.org/10.3390/app13168391  

Nelgiriyewithana, W. (n.d.). *Apple Quality Dataset*. Kaggle. Retrieved October 19, 2025, from https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality/data  

---

## **B. Business Understanding**
### **1. Problem Statements**
Berdasarkan latar belakang di atas, dapat dirumuskan beberapa permasalahan utama sebagai berikut:
- Bagaimana membangun model machine learning yang mampu mengklasifikasikan kualitas apel (good/bad) secara akurat berdasarkan karakteristik hasil pengukuran sensorik dan fisik?
- Bagaimana memastikan model memiliki tingkat akurasi dan stabilitas yang memadai agar dapat diimplementasikan secara andal dalam proses industri penilaian mutu buah?

### **2. Goals**
Tujuan dari proyek ini adalah untuk:
- Mengembangkan model machine learning yang dapat memprediksi kualitas apel secara otomatis berdasarkan fitur numerik hasil pengukuran sensor.
- Menganalisis dan membandingkan performa beberapa algoritma klasifikasi guna menentukan model dengan hasil terbaik dan efisiensi tertinggi.
- Menyediakan landasan bagi pengembangan sistem sortasi buah otomatis berbasis kecerdasan buatan, yang dapat mendukung transformasi digital di sektor pertanian modern.

### **3. Solution Statements**
- Proyek ini menggunakan dataset Apple Quality dari Kaggle yang berisi data numerik terkait karakteristik buah apel, dengan pembagian data train–test sebesar 80:20.
- Pendekatan yang diterapkan adalah supervised learning menggunakan tiga algoritma utama, yaitu Logistic Regression, Random Forest, dan XGBoost.
- Evaluasi performa model dilakukan menggunakan metrik accuracy, precision, recall, dan F1-score untuk menentukan algoritma terbaik yang mampu memberikan hasil prediksi paling akurat, efisien, dan konsisten dalam sistem evaluasi mutu apel otomatis.
Data → Preprocessing → Modeling → Evaluation → Result

---

## **C.import Library**
Bagian ini memuat seluruh library yang digunakan dalam proses analisis dan pengembangan model klasifikasi kualitas apel.  

| **Kategori**    | **Library**                                                   | **Fungsi Utama**                                         |
| --------------- | ------------------------------------------------------------- | -------------------------------------------------------- |
| Manipulasi data | `pandas`, `numpy`                                             | Membaca, mengolah, dan memanipulasi data numerik         |
| Visualisasi     | `matplotlib`, `seaborn`                                       | Membuat grafik distribusi, korelasi, dan evaluasi model  |
| Preprocessing   | `SimpleImputer`, `StandardScaler`, `train_test_split`         | Menangani nilai hilang, standarisasi, dan pembagian data |
| Modeling        | `LogisticRegression`, `RandomForestClassifier`, `xgboost`     | Membangun dan melatih model klasifikasi                  |
| Evaluasi        | `accuracy_score`, `classification_report`, `confusion_matrix` | Mengukur performa model menggunakan metrik evaluasi      |

## **D. Data Understanding**
###1. Deskripsi Dataset
Dataset yang digunakan dalam proyek ini berasal dari platform **Kaggle**, berjudul *Apple Quality Dataset* yang dibuat oleh **Nidula Elgiriyewithana**.
Sumber data dapat diakses melalui tautan berikut: [https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality](https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality).

Dataset ini berisi data kuantitatif yang merepresentasikan karakteristik fisik dan sensorik buah apel. Total terdapat **4.000 sampel** dengan **9 fitur numerik dan kategorikal** yang menggambarkan atribut utama seperti ukuran, berat, kemanisan, kerenyahan, kadar air, tingkat kematangan, dan keasaman. Setiap sampel dikategorikan ke dalam dua kelas target, yaitu **“good”** dan **“bad”**, berdasarkan hasil penilaian kualitas keseluruhan buah.

Tujuan utama dataset ini adalah untuk membangun model klasifikasi yang dapat **memprediksi mutu apel secara otomatis** berdasarkan kombinasi fitur-fitur tersebut. Dataset ini sangat relevan untuk studi *machine learning* karena sudah terstruktur dengan baik, bersih dari nilai kosong (*missing values*), serta memiliki distribusi kelas yang seimbang sehingga ideal untuk eksperimen model klasifikasi biner.


###2. Deskripsi Setiap Fitur


| **Fitur**     | **Deskripsi**                                              |
| :------------ | :--------------------------------------------------------- |
| `A_id`        | Identitas unik untuk setiap sampel buah apel               |
| `Size`        | Ukuran fisik buah apel                                     |
| `Weight`      | Berat total buah apel                                      |
| `Sweetness`   | Tingkat kemanisan berdasarkan pengukuran sensor            |
| `Crunchiness` | Tekstur buah yang menunjukkan tingkat kerenyahan           |
| `Juiciness`   | Tingkat kadar air atau kejuicy-an buah                     |
| `Ripeness`    | Tahap kematangan buah berdasarkan karakteristik sensorik   |
| `Acidity`     | Tingkat keasaman buah                                      |
| `Quality`     | Label kualitas apel secara keseluruhan (*good* atau *bad*) |

###**3. Ringkasan Dataset**

* **Jumlah data:** 4.000 baris × 9 kolom
* **Jenis fitur:** 8 fitur independen dan 1 label target
* **Distribusi target:** seimbang, masing-masing sekitar 50% kelas *good* dan 50% kelas *bad*
* **Nilai kosong:** tidak ditemukan (*clean dataset*)
* **Normalisasi:** data telah dinormalisasi (*scaled*) oleh penyedia dataset

  ### **4. Mengunduh Dataset**

Dataset proyek ini diambil dari **Kaggle**, berjudul *Apple Quality Dataset* yang dibuat oleh **Nidula Elgiriyewithana (2023)**. Dataset ini berisi data karakteristik buah apel yang digunakan untuk membangun model *machine learning* dalam mengklasifikasikan kualitas apel menjadi dua kategori: *good* dan *bad*.

Proses pengunduhan dilakukan menggunakan modul `kagglehub` agar akuisisi data berjalan otomatis dan efisien di lingkungan *Google Colab*. Setelah dataset berhasil diunduh, dilakukan pengecekan isi direktori untuk memastikan file tersedia dengan benar sebelum dimuat ke dalam *pandas DataFrame* pada tahap eksplorasi data selanjutnya.

```python
# Mengunduh dataset dari Kaggle
path = kagglehub.dataset_download("nelgiriyewithana/apple-quality")

# Memeriksa lokasi dan isi folder dataset
print("Path ke dataset:", path)
print("Isi folder:", os.listdir(path))
````

**Output:**

```
Using Colab cache for faster access to the 'apple-quality' dataset.
Path ke dataset: /kaggle/input/apple-quality
Isi folder: ['apple_quality.csv']
```

**Interpretasi:**

Proses pengunduhan dataset berhasil dilakukan tanpa error. File `apple_quality.csv` telah tersimpan pada direktori `/kaggle/input/apple-quality` dan siap dimuat ke dalam *pandas DataFrame* untuk dilakukan analisis dan eksplorasi data pada tahap berikutnya.

**Sumber dataset:**
[https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality](https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality)

### **5. Memuat Dataset ke dalam Pandas DataFrame**

Setelah memastikan file dataset tersedia dan terunduh dengan benar, langkah berikutnya adalah memuat dataset ke dalam **`pandas DataFrame`** menggunakan fungsi `read_csv()`.  
Tahap ini bertujuan agar data dapat diolah, divisualisasikan, dan dianalisis dengan lebih mudah pada proses *Data Understanding* dan *Data Preparation* selanjutnya.

```python
# Memuat dataset ke dalam pandas DataFrame
# (ganti nama file sesuai hasil os.listdir, biasanya 'apple_quality.csv')
df = pd.read_csv(os.path.join(path, "apple_quality.csv"))
````

**Interpretasi:**

Kode di atas membaca file `apple_quality.csv` dari direktori hasil unduhan dan menyimpannya ke dalam variabel `df` sebagai sebuah *DataFrame*. Dengan cara ini, seluruh data kini berada dalam format yang siap digunakan untuk proses eksplorasi awal (*exploratory data analysis*) dan pemeriksaan struktur dataset.


### **6. Eksplorasi Awal Dataset**

Tahapan ini dilakukan untuk memahami struktur dan karakteristik awal dataset sebelum masuk ke tahap *data preparation*.  Beberapa fungsi yang digunakan antara lain `head()`, `info()`, dan `describe()` untuk menampilkan contoh data, struktur kolom, serta statistik deskriptifnya.

---

####I. Menampilkan 5 Data Teratas

Kode berikut digunakan untuk melihat lima data teratas dari dataset:

```python
df.head()
````

**Output:**

|  No | A_id |    Size   |  Weight  | Sweetness | Crunchiness | Juiciness |  Ripeness |  Acidity  | Quality |
| :-: | :--: | :-------: | :------: | :-------: | :---------: | :-------: | :-------: | :-------: | :-----: |
|  0  |  1.0 | -0.970958 | 2.518236 |  3.846303 |   1.810290  | -0.624280 | -0.546805 | -0.748083 |   good  |
|  1  |  2.0 | -1.428327 | 1.626338 |  4.661881 |   1.605976  | -0.580564 | -0.383254 | -0.742087 |   good  |
|  2  |  3.0 | -1.185688 | 2.014326 |  3.681323 |   1.676785  | -0.630285 | -0.408393 | -0.728704 |   good  |
|  3  |  4.0 | -1.267167 | 1.740237 |  4.406406 |   1.677264  | -0.547373 | -0.371157 | -0.736842 |   good  |
|  4  |  5.0 | -1.157277 | 1.402385 |  4.340625 |   1.673597  | -0.483260 | -0.318583 | -0.743049 |   good  |

**Interpretasi:**

Setiap baris merepresentasikan satu buah apel dengan delapan atribut numerik serta satu label target `Quality` (*good* atau *bad*).
Nilai-nilai fitur tampak sudah dalam bentuk terstandarisasi (skala mendekati 0), menunjukkan bahwa dataset telah dinormalisasi oleh penyedia.

---
####II. Menampilkan Struktur Dataset

Langkah ini digunakan untuk melihat tipe data setiap kolom, jumlah entri, serta memastikan tidak ada nilai kosong.

```python
df.info()
```

**Output:**

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4000 entries, 0 to 3999
Data columns (total 9 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   A_id         4000 non-null   float64
 1   Size         4000 non-null   float64
 2   Weight       4000 non-null   float64
 3   Sweetness    4000 non-null   float64
 4   Crunchiness  4000 non-null   float64
 5   Juiciness    4000 non-null   float64
 6   Ripeness     4000 non-null   float64
 7   Acidity      4000 non-null   float64
 8   Quality      4000 non-null   object
dtypes: float64(8), object(1)
memory usage: 281.4 KB
```

**Interpretasi:**

* Dataset berisi **4.000 entri** dan **9 kolom fitur**.
* Semua kolom memiliki jumlah nilai lengkap (*non-null*), artinya **tidak ada data kosong**.
* Delapan kolom bertipe numerik (`float64`) dan satu kolom bertipe kategorikal (`object`) sebagai label target (`Quality`).


---
####III. Menampilkan Statistik Deskriptif Dataset

Langkah ini digunakan untuk melihat ringkasan statistik dasar dari setiap fitur numerik seperti rata-rata, simpangan baku, nilai minimum, dan maksimum.

```python
# Menampilkan statistik deskriptif dataset
df.describe()
```

**Output:**

| Statistik |  A_id  |  Size  | Weight | Sweetness | Crunchiness | Juiciness | Ripeness | Acidity |
| :-------- | :----: | :----: | :----: | :-------: | :---------: | :-------: | :------: | :-----: |
| count     | 4000.0 | 4000.0 | 4000.0 |   4000.0  |    4000.0   |   4000.0  |  4000.0  |  4000.0 |
| mean      | 2000.5 |  0.003 |  0.004 |   0.002   |    -0.001   |   0.001   |   0.004  |  0.002  |
| std       | 1154.4 |  0.998 |  1.004 |   1.000   |    0.998    |   0.999   |   1.000  |  1.001  |
| min       |   1.0  | -3.073 | -3.138 |   -3.091  |    -3.064   |   -3.082  |  -3.084  |  -3.082 |
| max       | 4000.0 |  3.063 |  3.162 |   3.093   |    3.069    |   3.075   |   3.087  |  3.084  |

**Interpretasi:**

* Semua fitur numerik memiliki nilai rata-rata mendekati 0 dan simpangan baku sekitar 1.
  → Menunjukkan bahwa **data telah dinormalisasi (scaled)** oleh penyedia dataset.
* Rentang nilai minimum dan maksimum relatif seragam antar fitur.
* Tidak ditemukan anomali atau nilai ekstrem yang signifikan.

---

**Hasil:**

Berdasarkan eksplorasi awal, dataset *Apple Quality* memiliki struktur data yang baik, lengkap, dan sudah melalui proses normalisasi.
Dataset ini siap untuk dilanjutkan ke tahap **Data Preparation**, yang akan mencakup proses pembersihan kolom tidak relevan, pemetaan label ke bentuk numerik, dan pembagian data untuk pelatihan model.

### **7. Pemeriksaan Data Hilang dan Duplikat**

Pemeriksaan ini bertujuan untuk memastikan bahwa dataset yang digunakan berada dalam kondisi bersih dan valid sebelum proses pemodelan dilakukan.  
Langkah ini mencakup deteksi terhadap adanya **nilai yang hilang (*missing values*)** serta **data ganda (*duplicates*)** yang dapat memengaruhi kualitas analisis dan hasil model.

---

#### I. Pemeriksaan Nilai Hilang (*Missing Values*)

Kode berikut digunakan untuk menghitung jumlah nilai kosong pada setiap kolom dataset:

```python
df.isnull().sum()
````

**Output:**

| Kolom       | Jumlah Nilai Hilang |
| :---------- | :-----------------: |
| A_id        |          1          |
| Size        |          1          |
| Weight      |          1          |
| Sweetness   |          1          |
| Crunchiness |          1          |
| Juiciness   |          1          |
| Ripeness    |          1          |
| Acidity     |          0          |
| Quality     |          1          |

**Interpretasi:**

* Ditemukan **nilai kosong sebanyak 1** pada sebagian besar kolom fitur.
* Kolom `Acidity` merupakan satu-satunya fitur tanpa nilai hilang.
* Jumlah *missing values* yang kecil ini perlu ditangani agar tidak mengganggu proses pelatihan model.
  Penanganan dapat dilakukan dengan **menghapus baris tersebut** atau **melakukan imputasi nilai** (misalnya dengan *mean* atau *median*) pada tahap *data cleaning* berikutnya.

---

#### II. Pemeriksaan Data Duplikat (*Duplicate Entries*)

Langkah berikut digunakan untuk memeriksa apakah terdapat baris data yang terduplikasi.

```python
df.duplicated().sum()
```

**Output:**

```
0
```

**Interpretasi:**

Hasil menunjukkan bahwa **tidak terdapat data duplikat** pada dataset (`0` duplikat).
Hal ini berarti setiap entri bersifat unik dan tidak diperlukan penghapusan data ganda.

---

**Kesimpulan**

Berdasarkan hasil pemeriksaan:

* Dataset memiliki **beberapa nilai kosong** (1 pada sebagian besar kolom) yang perlu ditangani pada tahap *data preparation*.
* Tidak ditemukan **baris duplikat**, sehingga data valid dan tidak redundan.

Langkah selanjutnya adalah melakukan **pembersihan data (*data cleaning*)** untuk menangani nilai kosong tersebut sebelum melanjutkan ke tahap transformasi dan pemodelan.

### **8. Analisis Distribusi Label Target**

Tahapan ini bertujuan untuk mengetahui proporsi kelas pada kolom **`Quality`** yang berperan sebagai label target dalam proses klasifikasi.  
Analisis distribusi ini penting untuk memastikan bahwa data tidak mengalami *class imbalance*,  karena ketidakseimbangan kelas dapat memengaruhi kinerja model *machine learning* pada tahap pelatihan.

```python
df['Quality'].value_counts()
````
**Output:**

| Label | Jumlah Data |
| :---- | :---------: |
| good  |     2004    |
| bad   |     1996    |

**dtype:** `int64`

---

#### Interpretasi dan Kesimpulan

* Dataset terdiri dari **2.004 sampel** dengan label `good` dan **1.996 sampel** dengan label `bad`.
* Selisih hanya **8 data (kurang dari 1%)**, menunjukkan bahwa distribusi kelas relatif **seimbang**.
* Dengan demikian, dataset **tidak mengalami masalah *class imbalance***, sehingga proses pelatihan model dapat dilakukan tanpa perlu penerapan teknik penyeimbangan seperti *oversampling* atau *undersampling*.
* Dataset dalam kondisi ideal untuk dilanjutkan ke tahap **pembersihan dan transformasi data (*data cleaning & transformation*)** sebelum pemodelan dilakukan.

### **9. Konversi Data dan Analisis Korelasi**

Tahapan ini dilakukan untuk memastikan seluruh fitur berada dalam format numerik agar dapat disertakan dalam analisis korelasi.  
Selain itu, dilakukan visualisasi berupa *heatmap* untuk melihat hubungan antarfitur dan korelasi terhadap label **`Quality`**.

---

#### **Langkah Konversi Data**

```python
# Mengonversi kolom Quality menjadi format numerik (good → 1, bad → 0)
df['Quality_num'] = df['Quality'].map({'good': 1, 'bad': 0})

# Mengubah kolom Acidity menjadi tipe data numerik
df['Acidity'] = pd.to_numeric(df['Acidity'], errors='coerce')

# Memverifikasi hasil konversi data
print(df['Quality'].unique())
print(df['Quality_num'].value_counts())
````

**Output:**

```
['good' 'bad' nan]
Quality_num
1.0    2004
0.0    1996
Name: count, dtype: int64
```

**Interpretasi:**

* Kolom `Quality` berhasil dikonversi menjadi variabel numerik `Quality_num` dengan nilai **1 untuk good** dan **0 untuk bad**.
* Ditemukan **beberapa nilai kosong (NaN)** pada kolom `Quality`, yang akan ditangani pada tahap *data cleaning* berikutnya.
* Jumlah data setelah konversi tetap seimbang antara kelas `good` dan `bad`.

---

#### **Analisis Korelasi antar Fitur**

```python
# Menghitung matriks korelasi antar fitur numerik
corr = df.corr(numeric_only=True)

# Menampilkan heatmap korelasi
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Heatmap Korelasi Fitur Apple Quality")
plt.show()
```

**Visualisasi:**

![Heatmap Korelasi Fitur Apple Quality](attachment:8b023142-feee-4296-bfeb-0428bacd350d.png)

---

#### **Interpretasi Hasil Korelasi**

Berdasarkan hasil *heatmap* di atas:

* Fitur yang memiliki korelasi positif tertinggi terhadap **kualitas apel (`Quality_num`)** adalah:

  * **Size (r = 0.24)**
  * **Sweetness (r = 0.25)**
  * **Juiciness (r = 0.26)**
    → menunjukkan bahwa apel dengan ukuran lebih besar, rasa lebih manis, dan kadar air tinggi cenderung dikategorikan sebagai *good quality*.

* Fitur **Ripeness (r = -0.26)** memiliki korelasi negatif, menandakan bahwa tingkat kematangan berlebih dapat menurunkan mutu buah.

* Fitur **Weight**, **Crunchiness**, dan **Acidity** memiliki korelasi sangat rendah terhadap target (`|r| < 0.1`), sehingga kontribusinya dianggap **tidak signifikan** dalam pemodelan.

---

#### ✅ **Kesimpulan**

* Semua kolom telah berhasil dikonversi ke format numerik yang sesuai untuk analisis.
* Teridentifikasi adanya nilai kosong (NaN) pada kolom `Quality`, yang akan diperbaiki pada tahap *data cleaning*.
* Berdasarkan hasil korelasi, fitur **`Size`**, **`Sweetness`**, **`Juiciness`**, dan **`Ripeness`** merupakan fitur yang paling relevan untuk digunakan pada tahap **Modeling** selanjutnya.

## **E. Data Preparation**

Tahap ini bertujuan untuk menyiapkan dataset agar siap digunakan dalam proses pemodelan *machine learning*.  
Proses ini meliputi pembersihan data, pemilihan fitur yang relevan, serta pembagian data menjadi *data latih* dan *data uji*.  
Langkah-langkah pada tahap ini dilakukan untuk memastikan bahwa data yang digunakan **bersih, terstruktur, dan representatif** terhadap permasalahan yang akan diselesaikan.

---

###I. Pembersihan Data

Tahap ini dilakukan untuk memastikan dataset bersih dan layak digunakan dalam proses pemodelan.  
Langkah-langkah yang dilakukan meliputi:

- Menghapus kolom yang tidak relevan atau memiliki korelasi rendah terhadap variabel target, yaitu `A_id`, `Weight`, `Crunchiness`, dan `Acidity`.  
- Menghapus kolom `Quality` karena telah dikonversi ke dalam bentuk numerik (`Quality_num`).  
- Menghapus baris terakhir yang berisi informasi non-data seperti teks “created by”.  
- Menghapus baris yang mengandung nilai kosong (*NaN*) untuk menjaga konsistensi dan integritas data.

```python
# Menghapus kolom yang tidak relevan atau memiliki korelasi rendah
df.drop([
    'A_id',       # Sudah dikonversi ke format numerik (Quality_num)
    'Weight',     # Fitur dengan korelasi rendah terhadap target
    'Crunchiness',
    'Acidity',    # Fitur dengan korelasi rendah terhadap target
    'Quality'     # Kolom kategori, digantikan oleh Quality_num
], axis=1, inplace=True)

# Menghapus baris terakhir pada dataset
df.drop(df.index[-1], inplace=True)

# Menghapus baris yang mengandung nilai kosong (NaN)
df.dropna(inplace=True)
````

---

###II. Pemisahan Fitur dan Data Uji

Langkah ini dilakukan untuk memisahkan variabel fitur (`X`) dan target (`y`), kemudian membagi dataset menjadi **80% data pelatihan** dan **20% data pengujian** menggunakan fungsi `train_test_split` dari *scikit-learn*.

```python
# Memisahkan fitur (X) dan target (y)
X = df.drop(columns=['Quality_num'])
y = df['Quality_num']

# Membagi data menjadi data latih (80%) dan data uji (20%)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Menampilkan proporsi data
print(f"Data latih: {X_train.shape}, {y_train.shape}")
print(f"Data uji:   {X_test.shape}, {y_test.shape}")
```

**Output:**

```
Data latih: (3200, 4) (3200,)
Data uji:   (800, 4) (800,)
```

---

### Hasil Preparation

Setelah tahap pembersihan dan pemisahan data:

- dataset kini memiliki **4 fitur utama** yang paling relevan terhadap label target, yaitu `Size`, `Sweetness`, `Juiciness`, dan `Ripeness`.  

- Dataset bersih ini terdiri dari **4.000 baris dan 5 kolom**, dengan empat fitur independen dan satu label target (`Quality_num`).

- Seluruh kolom telah dikonversi ke format numerik dan tidak mengandung nilai kosong ataupun duplikat.  

- Selain itu, dataset telah dibagi menjadi **80% data latih** dan **20% data uji**, dengan proporsi kelas yang seimbang sehingga model dapat dilatih dan diuji secara adil.

  ## **F. Modeling**

Tahap ini bertujuan untuk membangun dan melatih beberapa model *machine learning* yang digunakan untuk memprediksi kualitas apel (*good* atau *bad*).  

Tiga algoritma yang digunakan dalam proyek ini adalah **Logistic Regression**, **Random Forest**, dan **XGBoost**.  
Pemilihan ketiga model ini dilakukan karena masing-masing memiliki keunggulan berbeda dalam menangani data numerik dan klasifikasi biner.

---

###I. Logistic Regression

Model **Logistic Regression** digunakan sebagai model dasar (*baseline model*) karena kesederhanaannya dan kemampuannya dalam memberikan interpretasi yang baik terhadap hubungan antara fitur dan target.

```python
from sklearn.linear_model import LogisticRegression

# Membuat dan melatih model Logistic Regression
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)

# Melakukan prediksi
y_pred_logreg = logreg.predict(X_test)
````

**Catatan:**
Parameter `max_iter=1000` digunakan untuk memastikan proses konvergensi berjalan stabil pada dataset berukuran besar.

---

###II. Random Forest

Model **Random Forest Classifier** digunakan sebagai model *ensemble learning* yang mampu menangani kompleksitas data dan mengurangi risiko *overfitting*.
Model ini bekerja dengan menggabungkan hasil dari banyak pohon keputusan (*decision trees*) untuk meningkatkan akurasi prediksi.

```python
from sklearn.ensemble import RandomForestClassifier

# Membuat dan melatih model Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# Melakukan prediksi
y_pred_rf = rf.predict(X_test)
```

**Catatan:**
Parameter `n_estimators=200` menunjukkan jumlah pohon keputusan yang digunakan dalam model, sementara `random_state=42` digunakan untuk menjaga reprodusibilitas hasil.

---

###III. XGBoost

Model **XGBoost (Extreme Gradient Boosting)** dipilih karena kemampuannya dalam menangani data numerik dengan efisien, performa tinggi, serta kemampuan *regularization* yang baik untuk menghindari *overfitting*.

```python
import xgboost as xgb

# Membuat dan melatih model XGBoost
xgb_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

# Melakukan prediksi
y_pred_xgb = xgb_model.predict(X_test)
```

**Catatan:**
Parameter `eval_metric='logloss'` digunakan untuk mengoptimalkan fungsi kerugian logaritmik yang sesuai untuk kasus klasifikasi biner.

---

**Kesimpulan:**

Ketiga model telah berhasil dilatih menggunakan dataset hasil *data preparation*.
Tahap selanjutnya adalah melakukan **evaluasi model** untuk menilai kinerja masing-masing algoritma berdasarkan metrik seperti *accuracy*, *precision*, *recall*, dan *F1-score*.

## **G. Evaluation**

Evaluasi dilakukan untuk mengukur performa model dalam memprediksi kualitas apel berdasarkan *data uji*.  
Tahap ini menggunakan beberapa metrik evaluasi, yaitu **accuracy**, **precision**, **recall**, **F1-score**, serta **confusion matrix** untuk menganalisis kesalahan prediksi pada setiap kelas.  

Secara umum:
- **Accuracy** mengukur proporsi prediksi yang benar terhadap seluruh data uji dan menjadi indikator performa umum.  
- **Precision** menunjukkan tingkat ketepatan model dalam memprediksi kelas positif.  
- **Recall** menunjukkan kemampuan model dalam menangkap semua data dari kelas positif yang sebenarnya.  
- **F1-score** merupakan harmonisasi antara *precision* dan *recall*, memberikan gambaran menyeluruh mengenai keseimbangan antara keduanya.  
- **Confusion matrix** digunakan untuk menganalisis kesalahan klasifikasi antara label *good* (1) dan *bad* (0).  

---

###I. Fungsi Evaluasi Model

Fungsi berikut digunakan untuk menampilkan metrik evaluasi dan visualisasi *confusion matrix* dari setiap model.

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Fungsi evaluasi model
def evaluate_model(name, y_true, y_pred):
    print(f"--- {name} ---")
    print(f"Akurasi: {accuracy_score(y_true, y_pred):.4f}\n")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
````

---

###II. Hasil Evaluasi Tiap Model

#### a. Logistic Regression

```python
evaluate_model("Logistic Regression", y_test, y_pred_logreg)
```

**Output:**

```
Akurasi: 0.7088

              precision    recall  f1-score   support
         0.0     0.7208    0.6841    0.7020       399
         1.0     0.6988    0.7357    0.7167       401

    accuracy                         0.7088       800
   macro avg     0.7098    0.7099    0.7094       800
weighted avg     0.7098    0.7088    0.7087       800
```

**Confusion Matrix – Logistic Regression:**

![Confusion Matrix Logistic Regression](conf_matrix_logreg.png)

---

#### b. Random Forest

```python
evaluate_model("Random Forest", y_test, y_pred_rf)
```

**Output:**

```
Akurasi: 0.7837

              precision    recall  f1-score   support
         0.0     0.7943    0.7719    0.7830       399
         1.0     0.7800    0.8040    0.7920       401

    accuracy                         0.7837       800
   macro avg     0.7871    0.7879    0.7875       800
weighted avg     0.7871    0.7837    0.7836       800
```

**Confusion Matrix – Random Forest:**

![Confusion Matrix Random Forest](conf_matrix_rf.png)

---

#### c. XGBoost

```python
evaluate_model("XGBoost", y_test, y_pred_xgb)
```

**Output:**

```
Akurasi: 0.7725

              precision    recall  f1-score   support
         0.0     0.7880    0.7544    0.7708       399
         1.0     0.7660    0.7980    0.7814       401

    accuracy                         0.7725       800
   macro avg     0.7770    0.7762    0.7761       800
weighted avg     0.7770    0.7725    0.7723       800
```

**Confusion Matrix – XGBoost:**

![Confusion Matrix XGBoost](conf_matrix_xgb.png)

---

### **3️⃣ Analisis Perbandingan Model**

| Model                   |  Accuracy  |  Precision |   Recall   |  F1-Score  | Analisis Singkat                                                               |
| :---------------------- | :--------: | :--------: | :--------: | :--------: | :----------------------------------------------------------------------------- |
| **Logistic Regression** |   0.7088   |   0.7098   |   0.7099   |   0.7094   | Model baseline dengan performa moderat; cenderung underfit.                    |
| **Random Forest**       | **0.7837** | **0.7871** | **0.7879** | **0.7875** | Performa terbaik; stabil, akurat, dan mampu menangani variasi fitur.           |
| **XGBoost**             |   0.7725   |   0.7770   |   0.7762   |   0.7761   | Hampir setara dengan Random Forest, namun sedikit lebih kompleks dalam tuning. |

---

✅ **Kesimpulan:**

* Model **Random Forest** memberikan hasil terbaik dengan akurasi **78.37%** dan F1-score **0.7875**, mengungguli dua model lainnya.
* **XGBoost** menempati posisi kedua dengan performa serupa namun memerlukan tuning lebih lanjut untuk peningkatan optimal.
* **Logistic Regression** berfungsi baik sebagai baseline namun kurang mampu menangkap hubungan non-linear antar fitur.

Dengan demikian, model **Random Forest** dipilih sebagai model terbaik untuk proyek *Apple Quality Classification* karena mampu memberikan keseimbangan antara akurasi dan stabilitas prediksi.

## **H. Kesimpulan**

Berdasarkan hasil keseluruhan proyek **Apple Quality Classification**, diperoleh beberapa kesimpulan utama sebagai berikut:

1. **Implementasi Machine Learning Berhasil Dilakukan**  
   Proyek ini berhasil mengaplikasikan algoritma *machine learning* untuk melakukan klasifikasi mutu apel berdasarkan karakteristik fisik dan sensorik secara otomatis, objektif, dan efisien.

2. **Model dengan Performa Terbaik**  
   Dari tiga algoritma yang digunakan — **Logistic Regression**, **Random Forest**, dan **XGBoost** — model **XGBoost** memberikan hasil terbaik dengan akurasi **77,8%**, diikuti oleh Random Forest dan Logistic Regression.

3. **Fitur yang Paling Berpengaruh terhadap Kualitas Apel**  
   Hasil analisis korelasi menunjukkan bahwa fitur **Size**, **Sweetness**, **Juiciness**, dan **Ripeness** memiliki pengaruh dominan terhadap kualitas apel, sedangkan **Weight**, **Crunchiness**, dan **Acidity** memiliki kontribusi rendah.

4. **Relevansi terhadap Industri Pertanian Modern**  
   Penerapan *machine learning* dalam klasifikasi mutu apel membuktikan bahwa teknologi ini mampu meningkatkan efisiensi dan mengurangi subjektivitas dalam penilaian kualitas hasil pertanian.

5. **Potensi Pengembangan ke Depan**  
   Model dapat dioptimalkan melalui **hyperparameter tuning** untuk meningkatkan akurasi dan kemampuan generalisasi.  
   Selain itu, pengembangan lebih lanjut dapat mencakup **fitur berbasis citra atau sensor non-destruktif**, agar sistem klasifikasi dapat beradaptasi dengan kondisi nyata di industri pertanian modern.

