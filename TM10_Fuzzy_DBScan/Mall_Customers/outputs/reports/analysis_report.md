# Laporan Analisis Praktikum Machine Learning: Clustering Mall Customers

## 1. Judul Praktikum
Perbandingan Kinerja Algoritma K-Means, Fuzzy C-Means, dan DBSCAN pada Dataset Mall Customers.

## 2. Tujuan
1. Menerapkan algoritma clustering alternatif (Fuzzy C-Means dan DBSCAN) pada dataset Mall Customers.
2. Mengevaluasi dan membandingkan kualitas hasil clustering menggunakan metrik internal (Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index, Runtime, dan persentase noise).
3. Menganalisis apakah metode alternatif memberikan hasil yang lebih baik dibandingkan baseline K-Means.

## 3. Deskripsi Dataset
Dataset **Mall_Customers** terdiri dari 200 baris data pelanggan dengan 5 atribut:
- `CustomerID`: Identifikasi unik pelanggan.
- `Gender`: Jenis kelamin pelanggan.
- `Age`: Usia pelanggan.
- `Annual Income (k$)`: Pendapatan tahunan pelanggan dalam ribu dolar.
- `Spending Score (1-100)`: Skor pengeluaran pelanggan yang ditentukan oleh mal berdasarkan perilaku belanja.

## 4. Preprocessing
- Fitur utama yang digunakan untuk clustering adalah `Annual Income (k$)` dan `Spending Score (1-100)`.
- Pengecekan *missing values* dilakukan (tidak ada missing value yang signifikan).
- Normalisasi fitur menggunakan `StandardScaler` agar setiap fitur memiliki mean 0 dan standard deviation 1, memastikan bobot jarak yang setara antar fitur.

## 5. Implementasi Fuzzy C-Means
- Menggunakan library `scikit-fuzzy` (`skfuzzy.cluster.cmeans`).
- Jumlah cluster $c = 5$, fuzzifier $m = 2.0$, error tolerance = $0.005$, maksimum iterasi = $1000$.
- Menghasilkan *membership matrix* di mana setiap sampel memiliki derajat keanggotaan pada setiap cluster, yang kemudian dikonversi menjadi *hard labels* menggunakan nilai keanggotaan maksimum (`argmax`).
- Partition Coefficient (FPC) tercatat pada model.

## 6. Implementasi DBSCAN
- Menggunakan `sklearn.cluster.DBSCAN`.
- Pemilihan parameter $\text{eps} = 0.5$ dan $\text{min_samples} = 5$ dibantu dengan analisis *k-distance plot* ($k=5$).
- DBSCAN mendeteksi 2 cluster valid dan mengidentifikasi 8 titik sebagai *noise* (label `-1`), dengan persentase noise sebesar 4.00%.

## 7. Baseline K-Means
- Menggunakan `sklearn.cluster.KMeans` dengan $k = 5$ (ditentukan berdasarkan metode *Elbow* dan *Silhouette Score* optimal).
- Membagi pelanggan menjadi 5 kelompok utama yang merepresentasikan segmen pelanggan (misalnya: high income - high spending, low income - low spending, dll.).

## 8. Metode Evaluasi
Karena *clustering* adalah pembelajaran tanpa pengawasan (*unsupervised learning*), evaluasi dilakukan menggunakan metrik internal:
1. **Silhouette Score**: Mengukur seberapa mirip suatu objek dengan clusternya sendiri dibandingkan dengan cluster lain (rentang [-1, 1], semakin tinggi semakin baik).
2. **Davies-Bouldin Index (DBI)**: Mengukur rasio jarak antar cluster terhadap ukuran di dalam cluster (semakin rendah semakin baik).
3. **Calinski-Harabasz Index (CH)**: Rasio dispersi antar cluster dan dalam cluster (semakin tinggi semakin baik).
4. **Runtime**: Waktu eksekusi komputasi.
5. **Noise Percentage**: Khusus untuk DBSCAN.

*Catatan Istilah Akurasi*: Dalam machine learning unsupervised, istilah "akurasi" klasifikasi tidak berlaku langsung; kualitas diukur melalui validitas struktur cluster (Silhouette, DBI, CH).

## 9. Hasil Eksperimen dan Tabel Perbandingan

| Metrik | K-Means | Fuzzy C-Means | DBSCAN |
|---|---:|---:|---:|
| Jumlah cluster | 5 | 5 | 2 |
| Silhouette Score | 0.5547 | 0.5547 | 0.3876 |
| Davies-Bouldin Index | 0.5722 | 0.5722 | 0.7889 |
| Calinski-Harabasz Index | 248.65 | 248.65 | 85.39 |
| Runtime (detik) | 0.0095 | 0.0043 | 0.0020 |
| Noise percentage | N/A | N/A | 4.00% |

## 10. Analisis dan Pembahasan Pertanyaan Tugas

1. **Apakah Fuzzy C-Means memberikan hasil yang lebih baik daripada K-Means?**
   Secara metrik internal (Silhouette, DBI, CH), Fuzzy C-Means menghasilkan skor yang hampir identik atau sangat mirip dengan K-Means ketika membership matrix dikonversi menjadi *hard labels* via argmax. Keunggulan utama FCM bukan pada peningkatan metrik *hard clustering*, melainkan kemampuannya memberikan informasi probabilitas/derajat keanggotaan (*soft clustering*) untuk pelanggan yang berada di batas antar cluster.

2. **Apakah DBSCAN memberikan hasil yang lebih baik daripada K-Means?**
   DBSCAN bekerja berdasarkan kepadatan data. Pada dataset Mall Customers, DBSCAN menghasilkan jumlah cluster yang berbeda dan mengidentifikasi beberapa titik sebagai noise. Tergantung pada parameter `eps`, performa metrik DBSCAN bisa sangat kompetitif, namun sangat sensitif terhadap pemilihan parameter `eps` dan `min_samples`.

3. **Metode mana yang mempunyai Silhouette Score tertinggi?**
   K-Means dan Fuzzy C-Means menunjukkan Silhouette Score yang sangat tinggi dan stabil karena asumsi partisi spherical pada 5 cluster alami dataset Mall Customers.

4. **Metode mana yang mempunyai Davies-Bouldin Index terendah?**
   K-Means dan Fuzzy C-Means memiliki Davies-Bouldin Index terendah (menunjukkan pemisahan cluster yang padat dan terpisah baik).

5. **Metode mana yang mempunyai Calinski-Harabasz Index tertinggi?**
   K-Means dan Fuzzy C-Means mencatat Calinski-Harabasz Index tertinggi di antara ketiga metode.

6. **Apakah hasil seluruh metrik menunjukkan metode terbaik yang sama?**
   Ya, untuk dataset Mall Customers dengan $k=5$, K-Means dan Fuzzy C-Means konsisten menjadi yang terbaik dalam hal kekompakan cluster spherical, sementara DBSCAN sangat dipengaruhi oleh titik noise.

7. **Bagaimana pengaruh jumlah cluster dan distribusi anggota terhadap interpretasi hasil?**
   Jumlah cluster $k=5$ membagi pelanggan secara intuitif (misal: Pendapatan rendah-Pengeluaran rendah, Pendapatan rendah-Pengeluaran tinggi, Pendapatan menengah-Pengeluaran menengah, Pendapatan tinggi-Pengeluaran rendah, Pendapatan tinggi-Pengeluaran tinggi). Jika jumlah cluster tidak disesuaikan dengan baik, interpretasi segmen pelanggan menjadi kabur.

8. **Apakah DBSCAN menghasilkan terlalu banyak noise?**
   Dengan parameter $\text{eps} = 0.5$ dan $\text{min_samples} = 5$, persentase noise tercatat sebesar 4.00%, yang masih dalam batas wajar dan mencerminkan titik-titik pencilan (*outliers*) di luar kelompok utama pelanggan.

9. **Apa kelebihan dan kekurangan masing-masing metode pada dataset Mall Customers?**
   - **K-Means**:
     - *Kelebihan*: Cepat, sederhana, menghasilkan cluster yang jelas dan terinterpretasi dengan baik.
     - *Kekurangan*: Sensitif terhadap outlier, harus menentukan $k$ di awal, mengasumsikan cluster berbentuk bola/spherical.
   - **Fuzzy C-Means**:
     - *Kelebihan*: Menyediakan informasi derajat keanggotaan (*soft clustering*), sangat bagus untuk data yang overlap/tumpang tindih.
     - *Kekurangan*: Lebih lambat dari K-Means, sensitif terhadap inisialisasi dan parameter $m$.
   - **DBSCAN**:
     - *Kelebihan*: Mampu mendeteksi bentuk cluster arbitrer dan mengidentifikasi noise/outlier secara otomatis tanpa harus menentukan jumlah cluster di awal.
     - *Kekurangan*: Sangat sensitif terhadap pemilihan $\text{eps}$ dan $\text{min_samples}$, serta kurang optimal jika densitas data bervariasi.

## 11. Kesimpulan
Berdasarkan eksperimen dan evaluasi metrik internal, **K-Means** dan **Fuzzy C-Means** ($k=5$) memberikan kualitas pemisahan cluster yang sangat baik dan stabil pada dataset Mall Customers. **Fuzzy C-Means** memberikan nilai tambah berupa analisis derajat keanggotaan pelanggan, sementara **DBSCAN** sangat efektif dalam mendeteksi *noise* atau pelanggan anomali namun memerlukan penyetelan parameter yang cermat.
