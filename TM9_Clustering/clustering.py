# import tools
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# folder output gambar + fungsi simpan (nama file = nama diagram)
os.makedirs('gambar', exist_ok=True)
def simpan(nama):
    plt.savefig(os.path.join('gambar', nama + '.png'), dpi=150, bbox_inches='tight')

# import data
df = pd.read_csv('adult.csv')
print(df.head())

# amati bentuk data
print(df.shape)

# Melihat ringkasan statistik deskriptif dari DataFrame
print(df.describe())                      # kolom numerik
print(df.describe(exclude=np.number))     # kolom kategorik

# cek null data
# Di dataset Adult, missing value ditulis sebagai '?' (bukan NaN) -> ubah ke NaN dulu
df = df.replace('?', np.nan)
print(df.isnull().sum())

# cek & hapus duplikat
print('Duplikat:', df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)
print('Shape setelah hapus duplikat:', df.shape)

# cek outlier
fitur_num = ['age', 'capital-gain', 'hours-per-week']
plt.figure('01_boxplot_outlier_awal', figsize=(15, 4))
for i, col in enumerate(fitur_num, 1):
    plt.subplot(1, 3, i)                  # satu subplot per fitur (skala berbeda)
    sns.boxplot(y=df[col])
    plt.title('Boxplot {}'.format(col))
plt.tight_layout()
simpan('01_boxplot_outlier_awal')
plt.show()

for col in fitur_num:                     # hitung outlier dengan metode IQR
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    n_out = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
    print('{}: {} outlier (IQR)'.format(col, n_out))

# handling outlier / skew (baris TIDAK dihapus)
# - capital-gain : log1p (sangat miring, ~90% bernilai 0, outlier valid secara domain)
# - age, hours-per-week : winsorizing persentil 1-99 (IQR terlalu sempit untuk fitur ini)
df['capital-gain-log'] = np.log1p(df['capital-gain'])
for col in ['age', 'hours-per-week']:
    lo, hi = df[col].quantile([0.01, 0.99])
    df[col + '-cap'] = df[col].clip(lo, hi)
    print('{}: dicap ke [{}, {}]'.format(col, lo, hi))

# perbandingan sebelum vs sesudah handling
plt.figure('02_boxplot_sebelum_vs_sesudah_handling', figsize=(15, 8))
pasangan = [('age', 'age-cap'),
            ('capital-gain', 'capital-gain-log'),
            ('hours-per-week', 'hours-per-week-cap')]
for i, (sebelum, sesudah) in enumerate(pasangan):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(y=df[sebelum])
    plt.title('Sebelum: {}'.format(sebelum))
    plt.subplot(2, 3, i + 4)
    sns.boxplot(y=df[sesudah])
    plt.title('Sesudah: {}'.format(sesudah))
plt.tight_layout()
simpan('02_boxplot_sebelum_vs_sesudah_handling')
plt.show()

# cek korelasi antar fitur numerik (setelah handling)
print(df[['age-cap', 'capital-gain-log', 'hours-per-week-cap']].corr())

# amati bentuk visual masing-masing fitur
plt.style.use('fivethirtyeight')
plt.figure('03_distplot_age_capitalgain_hours', figsize=(15, 6))
n = 0

for x in ['age', 'capital-gain', 'hours-per-week']:
    n += 1
    plt.subplot(1, 3, n)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    sns.histplot(df[x], kde=True, stat="density", kde_kws=dict(cut=3), bins=20)
    plt.title('Distplot of {}'.format(x))
plt.tight_layout()
simpan('03_distplot_age_capitalgain_hours')
plt.show()

# Ploting untuk mencari relasi antara age, capital-gain and hours-per-week
plt.figure('04_regplot_relasi_age_capitalgain_hours', figsize=(15, 12))
n = 0
for x in ['age', 'capital-gain', 'hours-per-week']:
    for y in ['age', 'capital-gain', 'hours-per-week']:
        n += 1
        plt.subplot(3, 3, n)
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        sns.regplot(x=x, y=y, data=df)
        plt.ylabel(y.split()[0] + ' ' + y.split()[1] if len(y.split()) > 1 else y)
plt.tight_layout()
simpan('04_regplot_relasi_age_capitalgain_hours')
plt.show()

# Melihat sebaran hours-per-week dan capital-gain pada gender
plt.figure('05_scatter_capitalgain_vs_hours_per_gender', figsize=(15, 8))
for gender in ['Male', 'Female']:
    plt.scatter(x='capital-gain', y='hours-per-week',
                data=df[df['gender'] == gender], s=20, alpha=0.5, label=gender)
plt.xlabel('capital-gain'), plt.ylabel('hours-per-week')
plt.title('capital-gain vs hours-per-week')
plt.legend()
plt.tight_layout()
simpan('05_scatter_capitalgain_vs_hours_per_gender')
plt.show()

# Merancang K-Means untuk capital-gain vs hours-per-week
# Menentukan nilai k yang sesuai dengan Elbow-Method
# pakai fitur hasil handling outlier, baru di-scaling
X1 = df[['capital-gain-log', 'hours-per-week-cap']].values
# K-Means berbasis jarak Euclidean -> samakan skala fitur (standardisasi)
scaler = StandardScaler()
X1 = scaler.fit_transform(X1)
inertia = []
for n in range(1, 11):
    algorithm = (KMeans(n_clusters=n, init='k-means++', n_init=10,
                        max_iter=300, random_state=111))
    algorithm.fit(X1)
    inertia.append(algorithm.inertia_)

# Plot bentuk visual elbow
plt.figure('06_elbow_method_kmeans', figsize=(15, 6))
plt.plot(range(1, 11), inertia, 'o')
plt.plot(range(1, 11), inertia, '-', alpha=0.5)
plt.xlabel('Number of Clusters'), plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.tight_layout()
simpan('06_elbow_method_kmeans')
plt.show()

# Membangun K-Means
algorithm = (KMeans(n_clusters=5,   # CEK ULANG nilai n sesuai grafik elbow terbaru (data sudah berubah)
                    init='k-means++', n_init=10, max_iter=300,
                    tol=0.0001, random_state=111, algorithm='elkan'))
algorithm.fit(X1)
labels2 = algorithm.labels_
centroids2 = algorithm.cluster_centers_
# centroid dalam satuan asli; capital-gain dibalikkan dari log1p dengan expm1
cent_asli = scaler.inverse_transform(centroids2)
cent_asli[:, 0] = np.expm1(cent_asli[:, 0])
print(pd.DataFrame(cent_asli, columns=['capital-gain', 'hours-per-week (cap)']))

# Menyiapkan data untuk bentuk visual cluster
labels2 = algorithm.labels_
centroids2 = algorithm.cluster_centers_
# step dibuat relatif terhadap rentang data (step tetap 0.02 membuat grid terlalu besar & crash)
step_x = (X1[:, 0].max() - X1[:, 0].min()) / 500
step_y = (X1[:, 1].max() - X1[:, 1].min()) / 500
x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1
y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, step_x), np.arange(y_min, y_max, step_y))
Z1 = algorithm.predict(np.c_[xx.ravel(), yy.ravel()])  # array diratakan 1D

# Melihat bentuk visual cluster
plt.figure('07_visualisasi_cluster_kmeans', figsize=(15, 7))
plt.clf()
Z1 = Z1.reshape(xx.shape)
plt.imshow(Z1, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Pastel2, aspect='auto', origin='lower')
plt.scatter(x=X1[:, 0], y=X1[:, 1], c=labels2, s=20)   # X1 sudah ter-scaling
plt.scatter(x=centroids2[:, 0], y=centroids2[:, 1], s=300, c='red', alpha=0.5)
plt.ylabel('hours-per-week-cap (scaled)'), plt.xlabel('capital-gain-log (scaled)')
plt.title('Visualisasi Cluster K-Means')
plt.tight_layout()
simpan('07_visualisasi_cluster_kmeans')
plt.show()

# Melihat nilai Silhouette Score
# sample_size dipakai karena data ~48 ribu baris (hitung penuh sangat lambat/berat memori)
score2 = silhouette_score(X1, labels2, sample_size=10000, random_state=111)
print("Silhouette Score: ", score2)

# =====================================================================
# K-PROTOTYPES (data campuran numerik + kategorik)
# =====================================================================
from kmodes.kprototypes import KPrototypes

# Menentukan fitur numerik dan kategorik
# (education dibuang karena sudah diwakili educational-num; fnlwgt = bobot sampel;
#  income = label/target, dipakai untuk interpretasi saja, bukan untuk clustering;
#  native-country & race sangat timpang (mayoritas 1 kategori) sehingga tidak dipakai)
# num_cols memakai fitur hasil handling outlier
num_cols = ['age-cap', 'capital-gain-log', 'hours-per-week-cap']
cat_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'gender']

df_kp = df[num_cols + cat_cols].copy()
df_kp[cat_cols] = df_kp[cat_cols].fillna('Unknown').astype(str)   # '?' (NaN) jadi kategori 'Unknown'

# Scaling hanya untuk fitur numerik (kategorik tidak di-scaling / di-encode)
scaler_kp = StandardScaler()
num_scaled = scaler_kp.fit_transform(df_kp[num_cols])

# Gabungkan numerik (ter-scaling) + kategorik -> array bertipe object
X_kp = pd.concat([pd.DataFrame(num_scaled, columns=num_cols, index=df_kp.index),
                  df_kp[cat_cols]], axis=1).values
cat_idx = list(range(len(num_cols), len(num_cols) + len(cat_cols)))   # posisi kolom kategorik

# Menentukan nilai k dengan Elbow-Method (cost K-Prototypes)
# Pakai sampel 10.000 baris supaya cepat (K-Prototypes jauh lebih lambat dari K-Means)
rs = np.random.RandomState(111)
idx_sample = rs.choice(len(X_kp), size=min(10000, len(X_kp)), replace=False)
X_sample = X_kp[idx_sample]

cost_kp = []
for k in range(2, 9):
    kp = KPrototypes(n_clusters=k, init='Huang', n_init=1, max_iter=100, random_state=111)
    kp.fit(X_sample, categorical=cat_idx)
    cost_kp.append(kp.cost_)

# Plot bentuk visual elbow
plt.figure('08_elbow_method_kprototypes', figsize=(15, 6))
plt.plot(range(2, 9), cost_kp, 'o')
plt.plot(range(2, 9), cost_kp, '-', alpha=0.5)
plt.xlabel('Number of Clusters'), plt.ylabel('Cost')
plt.title('Elbow Method K-Prototypes')
plt.tight_layout()
simpan('08_elbow_method_kprototypes')
plt.show()

# Membangun K-Prototypes pada seluruh data
k_opt = 4   # CEK ULANG sesuai hasil elbow di atas (data sudah berubah)
kproto = KPrototypes(n_clusters=k_opt, init='Huang', n_init=3, max_iter=100, random_state=111)
labels_kp = kproto.fit_predict(X_kp, categorical=cat_idx)
print("Gamma (bobot kategorik): ", kproto.gamma)
print("Cost: ", kproto.cost_)

# Melihat profil tiap cluster
# rata-rata numerik diambil dari kolom asli di df supaya terbaca dalam satuan asli
df_kp['cluster'] = labels_kp
df['cluster_kp'] = labels_kp                      # index df dan df_kp sama
print(df_kp['cluster'].value_counts().sort_index())                                  # ukuran cluster
print(df.groupby('cluster_kp')[['age', 'capital-gain', 'hours-per-week']].mean())    # rata-rata numerik (satuan asli)
print(df_kp.groupby('cluster')[cat_cols].agg(lambda s: s.mode().iloc[0]))            # kategori dominan

# Melihat bentuk visual cluster (2 fitur numerik; cluster juga dibentuk oleh kategorik,
# jadi wajar bila terlihat saling tumpang tindih)
# ambil dari df (df_kp tidak lagi punya kolom capital-gain & hours-per-week)
plt.figure('09_visualisasi_cluster_kprototypes', figsize=(15, 7))
plt.scatter(x=df['capital-gain'], y=df['hours-per-week'], c=labels_kp, s=20, alpha=0.5)
plt.xlabel('capital-gain'), plt.ylabel('hours-per-week')
plt.title('Visualisasi Cluster K-Prototypes')
plt.tight_layout()
simpan('09_visualisasi_cluster_kprototypes')
plt.show()

# Melihat nilai Silhouette Score
# Euclidean tidak cocok untuk data campuran -> pakai jarak Gower pada sampel 3.000 baris
def gower_matrix(num, cat):
    n_row = len(num)
    rng_num = num.max(axis=0) - num.min(axis=0)
    rng_num[rng_num == 0] = 1
    d_num = np.zeros((n_row, n_row))
    for j in range(num.shape[1]):
        d_num += np.abs(num[:, j][:, None] - num[:, j][None, :]) / rng_num[j]
    d_cat = np.zeros((n_row, n_row))
    for j in range(cat.shape[1]):
        d_cat += (cat[:, j][:, None] != cat[:, j][None, :])
    return (d_num + d_cat) / (num.shape[1] + cat.shape[1])

idx_sil = rs.choice(len(df_kp), size=min(3000, len(df_kp)), replace=False)
num_sil = df_kp[num_cols].values[idx_sil].astype(float)
cat_sil = df_kp[cat_cols].apply(lambda s: s.astype('category').cat.codes).values[idx_sil]
dist_gower = gower_matrix(num_sil, cat_sil)
score_kp = silhouette_score(dist_gower, labels_kp[idx_sil], metric='precomputed')
print("Silhouette Score (Gower): ", score_kp)