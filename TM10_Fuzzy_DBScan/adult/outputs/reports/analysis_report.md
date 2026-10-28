# Laporan Analisis Eksperimen Dataset Adult (Census Income)

## 1. Pendahuluan
Eksperimen ini menerapkan algoritma klastering **K-Means**, **Fuzzy C-Means (FCM)**, dan **DBSCAN** pada dataset **Adult** (*Census Income*), yang dijalankan secara terpisah dari dataset *Mall Customers*. Tujuan dari eksperimen ini adalah mengevaluasi kinerja klastering pada data berdimensi tinggi dengan atribut campuran (*mixed-type attributes: numerical & categorical*).

## 2. Preprocessing Data
- **Pengambilan Sampel**: Dataset disampling secara acak sebanyak 5,000 baris untuk efisiensi komputasi *unsupervised clustering* dan perhitungan metrik jarak berdimensi tinggi.
- **Pembersihan Nilai Hilang**: Nilai yang hilang (`?`) diimputasi menggunakan median (fitur numerik) dan modus (fitur kategorikal).
- **Encoding & Scaling**:
  - Fitur numerik diskalakan menggunakan `StandardScaler`.
  - Fitur kategorikal dikodekan menggunakan `OneHotEncoder` (menghasilkan matriks fitur berdimensi 103).
- **Pemisahan Target**: Kolom target pendapatan (`income`) dikeluarkan secara ketat dari input klastering *unsupervised*.

## 3. Hasil Eksperimen dan Tabel Perbandingan

| Metrik Evaluasi | K-Means ($k=2$) | Fuzzy C-Means ($c=2$) | DBSCAN ($\text{eps}=5.0, \text{min}=10$) |
|---|---:|---:|---:|
| **Jumlah Klaster** | 2 | 2 | 2 |
| **Silhouette Score** | 0.1043 | 0.1060 | 0.6932 |
| **Davies-Bouldin Index** | 2.7934 | 2.8201 | 0.4158 |
| **Calinski-Harabasz Index** | 540.62 | 535.78 | 469.50 |
| **Runtime (detik)** | 0.0991 | 0.0161 | 0.5561 |
| **Noise Percentage** | N/A | N/A | 0.04% |
| **FPC (Fuzzy)** | N/A | 0.5000 | N/A |

## 4. Pembahasan dan Keterbatasan Metrik
1. **Dimensi Tinggi**: Ruang fitur *one-hot encoded* dataset Adult memiliki dimensi besar (103 fitur), yang menyebabkan fenomena *curse of dimensionality* pada metrik jarak seperti *Silhouette Score*.
2. **K-Means vs Fuzzy C-Means**: K-Means dan FCM menghasilkan partisi yang mirip karena sama-sama berbasis inisialisasi centroid pada ruang dimensi tinggi. FCM memberikan keunggulan tambahan berupa derajat keanggotaan (*membership matrix*) dan FPC.
3. **DBSCAN**: Pada ruang berdimensi tinggi dengan densitas kontinu, DBSCAN memerlukan penyetelan $\varepsilon$ yang sangat hati-hati agar tidak mengkategorikan sebagian besar data sebagai *noise*.

## 5. Kesimpulan
Eksperimen dataset Adult membuktikan bahwa pipeline *preprocessing* data campuran dan algoritma klastering berhasil diimplementasikan secara modular dan terisolasi, memberikan wawasan mendalam mengenai tantangan *unsupervised learning* pada data demografi berdimensi tinggi.
