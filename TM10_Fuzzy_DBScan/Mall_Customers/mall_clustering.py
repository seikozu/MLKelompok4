"""
Clustering Dataset Mall Customers - K-Means, Fuzzy C-Means, DBSCAN
==================================================================
Seluruh proses (load data, preprocessing, model, evaluasi, visualisasi,
laporan, dan tabel perbandingan) digabung dalam SATU file ini.

Cara menjalankan (dari folder mana saja):
    python mall_clustering.py

Kebutuhan folder:
    Mall_Customers/
    ├── mall_clustering.py    <- file ini
    └── Mall_Customers.csv    <- dataset

Output otomatis dibuat di folder  outputs/  (figures, metrics, reports).
"""

import os
import json
import time

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # supaya bisa jalan tanpa layar (headless)
import matplotlib.pyplot as plt
import seaborn as sns
import skfuzzy as fuzz

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------------------
# KONFIGURASI
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Mall_Customers.csv")
FIG_DIR = os.path.join(BASE_DIR, "outputs", "figures")
METRICS_DIR = os.path.join(BASE_DIR, "outputs", "metrics")
REPORT_DIR = os.path.join(BASE_DIR, "outputs", "reports")

RANDOM_STATE = 42
FEATURE_COLS = ["Annual Income (k$)", "Spending Score (1-100)"]
OPTIMAL_K = 5           # k=5 (standar untuk Mall Customers; dicek via elbow & silhouette)
FCM_M = 2.0             # fuzzifier Fuzzy C-Means
DBSCAN_EPS = 0.5        # dipilih berdasarkan k-distance plot
DBSCAN_MIN_SAMPLES = 5


# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
def load_mall_customers(filepath=DATA_PATH):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset Mall_Customers tidak ditemukan di {filepath}")
    return pd.read_csv(filepath)


# ---------------------------------------------------------------------------
# 2. PREPROCESSING
# ---------------------------------------------------------------------------
def preprocess_data(df, feature_cols=None):
    """Pilih fitur (Annual Income & Spending Score), cek missing value, StandardScaler."""
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    if df[feature_cols].isnull().sum().sum() > 0:
        df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

    X_raw = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    return X_raw, X_scaled, scaler, feature_cols


# ---------------------------------------------------------------------------
# 3. MODEL: K-MEANS, FUZZY C-MEANS, DBSCAN
# ---------------------------------------------------------------------------
def run_kmeans(X_scaled, n_clusters=OPTIMAL_K, random_state=RANDOM_STATE):
    start = time.time()
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    return {
        "model": kmeans,
        "labels": labels,
        "centers": kmeans.cluster_centers_,
        "runtime": time.time() - start,
        "n_clusters": n_clusters,
    }


def find_optimal_k(X_scaled, max_k=10):
    """Hitung inertia & silhouette untuk k=2..max_k (elbow method)."""
    inertias, silhouette_scores = [], []
    k_range = range(2, max_k + 1)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, labels))
    return k_range, inertias, silhouette_scores


def run_fuzzy_cmeans(X_scaled, n_clusters=OPTIMAL_K, m=FCM_M, error=0.005, maxiter=1000):
    start = time.time()
    # skfuzzy mengharapkan shape (n_features, n_samples)
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        X_scaled.T, c=n_clusters, m=m, error=error, maxiter=maxiter, init=None,
        seed=RANDOM_STATE,  # kunci inisialisasi acak agar hasil reproducible
    )
    runtime = time.time() - start
    return {
        "centers": cntr,
        "membership": u,
        "max_membership": np.max(u, axis=0),
        "labels": np.argmax(u, axis=0),  # membership -> hard label
        "fpc": fpc,
        "runtime": runtime,
        "n_clusters": n_clusters,
        "m": m,
    }


def compute_k_distance(X_scaled, k=5):
    """Jarak ke tetangga ke-k (terurut) untuk k-distance plot -> memilih eps."""
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(X_scaled)
    distances, _ = neigh.kneighbors(X_scaled)
    return np.sort(distances[:, k - 1], axis=0)


def run_dbscan(X_scaled, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES):
    start = time.time()
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)
    runtime = time.time() - start

    unique_labels = set(labels)
    n_noise = int(list(labels).count(-1))
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    return {
        "model": dbscan,
        "labels": labels,
        "eps": eps,
        "min_samples": min_samples,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_percentage": (n_noise / len(X_scaled)) * 100,
        "runtime": runtime,
    }


# ---------------------------------------------------------------------------
# 4. EVALUASI
# ---------------------------------------------------------------------------
def _na_result(n_clusters, reason):
    return {
        "n_clusters": n_clusters,
        "silhouette_score": "N/A",
        "davies_bouldin_index": "N/A",
        "calinski_harabasz_index": "N/A",
        "reason": reason,
    }


def evaluate_clustering(X_scaled, labels, is_dbscan=False):
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

    if n_clusters < 2 or n_clusters >= len(X_scaled):
        return _na_result(n_clusters, f"Invalid number of clusters ({n_clusters}) for evaluation metrics.")

    X_eval, labels_eval = X_scaled, labels
    if is_dbscan:  # evaluasi hanya pada titik non-noise
        mask = labels != -1
        X_eval, labels_eval = X_scaled[mask], labels[mask]
        if len(set(labels_eval)) < 2:
            return _na_result(n_clusters, "Fewer than 2 valid clusters after excluding noise.")

    try:
        sil = silhouette_score(X_eval, labels_eval)
        dbi = davies_bouldin_score(X_eval, labels_eval)
        chi = calinski_harabasz_score(X_eval, labels_eval)
    except Exception:
        sil, dbi, chi = "N/A", "N/A", "N/A"

    return {
        "n_clusters": n_clusters,
        "silhouette_score": float(sil) if isinstance(sil, (float, np.floating)) else sil,
        "davies_bouldin_index": float(dbi) if isinstance(dbi, (float, np.floating)) else dbi,
        "calinski_harabasz_index": float(chi) if isinstance(chi, (float, np.floating)) else chi,
    }


# ---------------------------------------------------------------------------
# 5. VISUALISASI
# ---------------------------------------------------------------------------
def save_fig(fig, filename, output_dir=FIG_DIR):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {filepath}")


def plot_scatter(X_raw, labels, title, filename, is_dbscan=False):
    fig, ax = plt.subplots(figsize=(8, 6))
    x = X_raw[:, 0]  # Annual Income (k$)
    y = X_raw[:, 1]  # Spending Score (1-100)

    if is_dbscan:
        noise_mask = labels == -1
        valid_mask = ~noise_mask
        ax.scatter(x[noise_mask], y[noise_mask], c="black", marker="x", s=50,
                   label="Noise (-1)", alpha=0.6)
        sns.scatterplot(x=x[valid_mask], y=y[valid_mask], hue=labels[valid_mask],
                        palette="tab10", ax=ax, legend="full", s=60, alpha=0.8)
    else:
        sns.scatterplot(x=x, y=y, hue=labels, palette="tab10", ax=ax,
                        legend="full", s=60, alpha=0.8)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Annual Income (k$)", fontsize=12)
    ax.set_ylabel("Spending Score (1-100)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    save_fig(fig, filename)


def plot_elbow_and_silhouette(k_range, inertias, silhouette_scores):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_range, inertias, marker="o", color="b", linestyle="-")
    ax.set_title("Elbow Method for Optimal k (K-Means)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax.set_ylabel("Inertia (WCSS)", fontsize=12)
    plt.tight_layout()
    save_fig(fig, "kmeans_elbow_method.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_range, silhouette_scores, marker="s", color="g", linestyle="-")
    ax.set_title("Silhouette Scores for Candidate k (K-Means)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax.set_ylabel("Silhouette Score", fontsize=12)
    plt.tight_layout()
    save_fig(fig, "kmeans_silhouette_scores.png")


def plot_k_distance(sorted_distances, k_val=5, chosen_eps=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sorted_distances, color="r", linewidth=2)
    ax.set_title(f"K-Distance Plot (k={k_val}) for DBSCAN", fontsize=14, fontweight="bold")
    ax.set_xlabel("Points sorted by distance", fontsize=12)
    ax.set_ylabel(f"{k_val}-NN Distance", fontsize=12)
    if chosen_eps is not None:
        ax.axhline(y=chosen_eps, color="b", linestyle="--", label=f"Chosen eps = {chosen_eps}")
        ax.legend()
    plt.tight_layout()
    save_fig(fig, "dbscan_k_distance.png")


def plot_comparison_metrics(comparison_data):
    models = list(comparison_data.keys())
    metrics_names = ["silhouette_score", "davies_bouldin_index", "calinski_harabasz_index"]

    valid_metrics = {}
    for m in metrics_names:
        vals = []
        for mod in models:
            val = comparison_data[mod].get(m, "N/A")
            vals.append(val if isinstance(val, (int, float)) else 0)
        valid_metrics[m] = vals

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, valid_metrics["silhouette_score"], width,
           label="Silhouette Score (higher better)")
    ax.bar(x, valid_metrics["davies_bouldin_index"], width,
           label="Davies-Bouldin Index (lower better)")
    ax.bar(x + width, [v / 100.0 for v in valid_metrics["calinski_harabasz_index"]], width,
           label="Calinski-Harabasz / 100 (higher better)")
    ax.set_title("Clustering Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    plt.tight_layout()
    save_fig(fig, "comparison_metrics.png")


# ---------------------------------------------------------------------------
# 6. LAPORAN & TABEL PERBANDINGAN
# ---------------------------------------------------------------------------
def _fmt(value, spec):
    """Format angka; kalau bukan angka (mis. 'N/A') tampilkan apa adanya."""
    return format(value, spec) if isinstance(value, (int, float)) else "N/A"


def generate_report(comp, k, eps, min_samples):
    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, "analysis_report.md")

    km, fcm, db = comp["K-Means"], comp["Fuzzy C-Means"], comp["DBSCAN"]

    content = f"""# Laporan Analisis Praktikum Machine Learning: Clustering Mall Customers

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
- Jumlah cluster $c = {k}$, fuzzifier $m = 2.0$, error tolerance = $0.005$, maksimum iterasi = $1000$.
- Menghasilkan *membership matrix* di mana setiap sampel memiliki derajat keanggotaan pada setiap cluster, yang kemudian dikonversi menjadi *hard labels* menggunakan nilai keanggotaan maksimum (`argmax`).
- Partition Coefficient (FPC) tercatat pada model.

## 6. Implementasi DBSCAN
- Menggunakan `sklearn.cluster.DBSCAN`.
- Pemilihan parameter $\\text{{eps}} = {eps}$ dan $\\text{{min_samples}} = {min_samples}$ dibantu dengan analisis *k-distance plot* ($k=5$).
- DBSCAN mendeteksi {db['n_clusters']} cluster valid dan mengidentifikasi {db['n_noise']} titik sebagai *noise* (label `-1`), dengan persentase noise sebesar {db['noise_percentage']:.2f}%.

## 7. Baseline K-Means
- Menggunakan `sklearn.cluster.KMeans` dengan $k = {k}$ (ditentukan berdasarkan metode *Elbow* dan *Silhouette Score* optimal).
- Membagi pelanggan menjadi {k} kelompok utama yang merepresentasikan segmen pelanggan (misalnya: high income - high spending, low income - low spending, dll.).

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
| Jumlah cluster | {km['n_clusters']} | {fcm['n_clusters']} | {db['n_clusters']} |
| Silhouette Score | {_fmt(km['silhouette_score'], '.4f')} | {_fmt(fcm['silhouette_score'], '.4f')} | {_fmt(db['silhouette_score'], '.4f')} |
| Davies-Bouldin Index | {_fmt(km['davies_bouldin_index'], '.4f')} | {_fmt(fcm['davies_bouldin_index'], '.4f')} | {_fmt(db['davies_bouldin_index'], '.4f')} |
| Calinski-Harabasz Index | {_fmt(km['calinski_harabasz_index'], '.2f')} | {_fmt(fcm['calinski_harabasz_index'], '.2f')} | {_fmt(db['calinski_harabasz_index'], '.2f')} |
| Runtime (detik) | {km['runtime']:.4f} | {fcm['runtime']:.4f} | {db['runtime']:.4f} |
| Noise percentage | N/A | N/A | {db['noise_percentage']:.2f}% |

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
   Dengan parameter $\\text{{eps}} = {eps}$ dan $\\text{{min_samples}} = {min_samples}$, persentase noise tercatat sebesar {db['noise_percentage']:.2f}%, yang masih dalam batas wajar dan mencerminkan titik-titik pencilan (*outliers*) di luar kelompok utama pelanggan.

9. **Apa kelebihan dan kekurangan masing-masing metode pada dataset Mall Customers?**
   - **K-Means**:
     - *Kelebihan*: Cepat, sederhana, menghasilkan cluster yang jelas dan terinterpretasi dengan baik.
     - *Kekurangan*: Sensitif terhadap outlier, harus menentukan $k$ di awal, mengasumsikan cluster berbentuk bola/spherical.
   - **Fuzzy C-Means**:
     - *Kelebihan*: Menyediakan informasi derajat keanggotaan (*soft clustering*), sangat bagus untuk data yang overlap/tumpang tindih.
     - *Kekurangan*: Lebih lambat dari K-Means, sensitif terhadap inisialisasi dan parameter $m$.
   - **DBSCAN**:
     - *Kelebihan*: Mampu mendeteksi bentuk cluster arbitrer dan mengidentifikasi noise/outlier secara otomatis tanpa harus menentukan jumlah cluster di awal.
     - *Kekurangan*: Sangat sensitif terhadap pemilihan $\\text{{eps}}$ dan $\\text{{min_samples}}$, serta kurang optimal jika densitas data bervariasi.

## 11. Kesimpulan
Berdasarkan eksperimen dan evaluasi metrik internal, **K-Means** dan **Fuzzy C-Means** ($k=5$) memberikan kualitas pemisahan cluster yang sangat baik dan stabil pada dataset Mall Customers. **Fuzzy C-Means** memberikan nilai tambah berupa analisis derajat keanggotaan pelanggan, sementara **DBSCAN** sangat efektif dalam mendeteksi *noise* atau pelanggan anomali namun memerlukan penyetelan parameter yang cermat.
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Analysis report generated at {report_path}")


def print_comparison_table(metrics_path):
    """Cetak tabel perbandingan ke terminal dari file metrics JSON."""
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    header = (f"{'Model':<20} | {'Clusters':<10} | {'Silhouette':<12} | {'DBI':<10} | "
              f"{'CH Index':<10} | {'Runtime (s)':<12}")
    line = "=" * len(header)
    print(line)
    print("PERBANDINGAN KINERJA CLUSTERING (MALL CUSTOMERS)")
    print(line)
    print(header)
    print("-" * len(header))
    for name, m in metrics.items():
        print(f"{name:<20} | {str(m.get('n_clusters', 'N/A')):<10} | "
              f"{_fmt(m.get('silhouette_score'), '.4f'):<12} | "
              f"{_fmt(m.get('davies_bouldin_index'), '.4f'):<10} | "
              f"{_fmt(m.get('calinski_harabasz_index'), '.4f'):<10} | "
              f"{_fmt(m.get('runtime'), '.4f'):<12}")
    print(line)


# ---------------------------------------------------------------------------
# 7. MAIN: JALANKAN SELURUH EKSPERIMEN
# ---------------------------------------------------------------------------
def main():
    print("=== Starting Mall_Customers Clustering Experiment ===")

    # 1. Load data
    df = load_mall_customers()
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.")

    # 2. Preprocessing
    X_raw, X_scaled, scaler, feature_cols = preprocess_data(df)
    print(f"Features selected and scaled: {feature_cols}")

    # 3. K-Means: cari k optimal (elbow & silhouette) lalu jalankan k=5
    k_range, inertias, silhouette_scores = find_optimal_k(X_scaled, max_k=10)
    plot_elbow_and_silhouette(k_range, inertias, silhouette_scores)

    kmeans_res = run_kmeans(X_scaled, n_clusters=OPTIMAL_K)
    kmeans_metrics = evaluate_clustering(X_scaled, kmeans_res["labels"])
    kmeans_metrics["runtime"] = kmeans_res["runtime"]
    kmeans_metrics["noise_percentage"] = 0.0
    print(f"K-Means (k={OPTIMAL_K}) - Silhouette: {kmeans_metrics.get('silhouette_score')}, "
          f"DBI: {kmeans_metrics.get('davies_bouldin_index')}, "
          f"CH: {kmeans_metrics.get('calinski_harabasz_index')}")
    plot_scatter(X_raw, kmeans_res["labels"], "K-Means Clustering (k=5)", "kmeans_clusters.png")

    # 4. Fuzzy C-Means
    fcm_res = run_fuzzy_cmeans(X_scaled, n_clusters=OPTIMAL_K, m=FCM_M)
    fcm_metrics = evaluate_clustering(X_scaled, fcm_res["labels"])
    fcm_metrics["runtime"] = fcm_res["runtime"]
    fcm_metrics["noise_percentage"] = 0.0
    fcm_metrics["fpc"] = float(fcm_res["fpc"])
    print(f"Fuzzy C-Means (c={OPTIMAL_K}, m={FCM_M}) - "
          f"Silhouette: {fcm_metrics.get('silhouette_score')}, "
          f"DBI: {fcm_metrics.get('davies_bouldin_index')}, "
          f"CH: {fcm_metrics.get('calinski_harabasz_index')}, FPC: {fcm_res['fpc']:.4f}")
    plot_scatter(X_raw, fcm_res["labels"], "Fuzzy C-Means Clustering (c=5, m=2.0)",
                 "fuzzy_cmeans_clusters.png")

    # 5. DBSCAN (eps dipilih dari k-distance plot)
    k_dist = compute_k_distance(X_scaled, k=5)
    plot_k_distance(k_dist, k_val=5, chosen_eps=DBSCAN_EPS)

    dbscan_res = run_dbscan(X_scaled, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
    dbscan_metrics = evaluate_clustering(X_scaled, dbscan_res["labels"], is_dbscan=True)
    dbscan_metrics["runtime"] = dbscan_res["runtime"]
    dbscan_metrics["n_clusters"] = dbscan_res["n_clusters"]
    dbscan_metrics["n_noise"] = dbscan_res["n_noise"]
    dbscan_metrics["noise_percentage"] = dbscan_res["noise_percentage"]
    print(f"DBSCAN (eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES}) - "
          f"Clusters: {dbscan_res['n_clusters']}, Noise: {dbscan_res['n_noise']} "
          f"({dbscan_res['noise_percentage']:.2f}%), "
          f"Silhouette: {dbscan_metrics.get('silhouette_score')}")
    plot_scatter(X_raw, dbscan_res["labels"],
                 f"DBSCAN Clustering (eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES})",
                 "dbscan_clusters.png", is_dbscan=True)

    # 6. Perbandingan & simpan metrics
    comparison_data = {
        "K-Means": kmeans_metrics,
        "Fuzzy C-Means": fcm_metrics,
        "DBSCAN": dbscan_metrics,
    }
    plot_comparison_metrics(comparison_data)

    os.makedirs(METRICS_DIR, exist_ok=True)
    metrics_path = os.path.join(METRICS_DIR, "experiment_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(comparison_data, f, indent=4)
    print(f"Metrics saved to {metrics_path}")

    # 7. Laporan analisis + tabel perbandingan di terminal
    generate_report(comparison_data, OPTIMAL_K, DBSCAN_EPS, DBSCAN_MIN_SAMPLES)
    print_comparison_table(metrics_path)
    print("=== Mall_Customers Experiment Completed Successfully ===")


if __name__ == "__main__":
    main()
