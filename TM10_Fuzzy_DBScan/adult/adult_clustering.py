"""
Clustering Dataset Adult (Census Income) - K-Means, Fuzzy C-Means, DBSCAN
=========================================================================
Seluruh proses (load data, preprocessing, model, evaluasi, visualisasi,
laporan, dan tabel perbandingan) digabung dalam SATU file ini.

Cara menjalankan (dari folder mana saja):
    python adult_clustering.py

Kebutuhan folder:
    adult/
    ├── adult_clustering.py   <- file ini
    └── adult.csv             <- dataset

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
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder

sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------------------
# KONFIGURASI
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "adult.csv")
FIG_DIR = os.path.join(BASE_DIR, "outputs", "figures")
METRICS_DIR = os.path.join(BASE_DIR, "outputs", "metrics")
REPORT_DIR = os.path.join(BASE_DIR, "outputs", "reports")

SAMPLE_SIZE = 5000      # sampling acak agar komputasi clustering efisien
RANDOM_STATE = 42
N_CLUSTERS = 2          # k=2 -> sesuai 2 kelas income (<=50K vs >50K)
FCM_M = 2.0             # fuzzifier Fuzzy C-Means
DBSCAN_EPS = 5.0
DBSCAN_MIN_SAMPLES = 10

NUM_COLS = ["age", "fnlwgt", "educational-num", "capital-gain",
            "capital-loss", "hours-per-week"]


# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
def load_adult_data(filepath=DATA_PATH, sample_size=SAMPLE_SIZE,
                    random_state=RANDOM_STATE):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset adult tidak ditemukan di {filepath}")

    df = pd.read_csv(filepath)
    print(f"Original Adult dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.")

    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
        print(f"Sampled Adult dataset to {sample_size} rows for efficient unsupervised clustering.")
    return df


# ---------------------------------------------------------------------------
# 2. PREPROCESSING
# ---------------------------------------------------------------------------
def preprocess_adult_data(df):
    """
    - Mengeluarkan target 'income' dari input clustering.
    - Imputasi missing value ('?'): median (numerik), modus (kategorikal).
    - Scaling numerik dengan StandardScaler.
    - One-hot encoding fitur kategorikal.
    """
    df = df.copy()

    if "income" in df.columns:
        y_true = df["income"]
        X_df = df.drop(columns=["income"])
    else:
        y_true = None
        X_df = df

    num_cols = [c for c in NUM_COLS if c in X_df.columns]
    cat_cols = [c for c in X_df.columns if c not in NUM_COLS]

    X_df = X_df.replace("?", np.nan)
    for col in num_cols:
        X_df[col] = X_df[col].fillna(X_df[col].median())
    for col in cat_cols:
        mode_val = X_df[col].mode()
        X_df[col] = X_df[col].fillna(mode_val[0] if not mode_val.empty else "Unknown")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )
    X_scaled = preprocessor.fit_transform(X_df)
    return X_df, X_scaled, preprocessor, y_true


# ---------------------------------------------------------------------------
# 3. MODEL: K-MEANS, FUZZY C-MEANS, DBSCAN
# ---------------------------------------------------------------------------
def run_adult_kmeans(X_scaled, n_clusters=N_CLUSTERS, random_state=RANDOM_STATE):
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


def run_adult_fuzzy_cmeans(X_scaled, n_clusters=N_CLUSTERS, m=FCM_M,
                           error=0.01, maxiter=500):
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
        "labels": np.argmax(u, axis=0),
        "fpc": fpc,
        "runtime": runtime,
        "n_clusters": n_clusters,
        "m": m,
    }


def run_adult_dbscan(X_scaled, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES):
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


def evaluate_adult_clustering(X_scaled, labels, is_dbscan=False):
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

    if n_clusters < 2 or n_clusters >= len(X_scaled):
        return _na_result(n_clusters, f"Invalid number of clusters ({n_clusters}) for evaluation.")

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
# 5. VISUALISASI (proyeksi PCA 2D)
# ---------------------------------------------------------------------------
def save_adult_fig(fig, filename, output_dir=FIG_DIR):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Adult figure saved to {filepath}")


def plot_adult_pca_scatter(X_scaled, labels, title, filename, is_dbscan=False):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(9, 6))
    if is_dbscan:
        noise_mask = labels == -1
        valid_mask = ~noise_mask
        ax.scatter(X_pca[noise_mask, 0], X_pca[noise_mask, 1], c="black", marker="x",
                   s=40, label="Noise (-1)", alpha=0.5)
        sns.scatterplot(x=X_pca[valid_mask, 0], y=X_pca[valid_mask, 1], hue=labels[valid_mask],
                        palette="tab10", ax=ax, legend="full", s=50, alpha=0.8)
    else:
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette="tab10",
                        ax=ax, legend="full", s=50, alpha=0.8)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    save_adult_fig(fig, filename)


def plot_adult_comparison(comparison_data):
    models = list(comparison_data.keys())
    metrics_names = ["silhouette_score", "davies_bouldin_index"]

    valid_metrics = {}
    for m in metrics_names:
        vals = []
        for mod in models:
            val = comparison_data[mod].get(m, "N/A")
            vals.append(val if isinstance(val, (int, float)) else 0)
        valid_metrics[m] = vals

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width / 2, valid_metrics["silhouette_score"], width,
           label="Silhouette Score (higher better)")
    ax.bar(x + width / 2, valid_metrics["davies_bouldin_index"], width,
           label="Davies-Bouldin Index (lower better)")
    ax.set_title("Adult Dataset Clustering Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    plt.tight_layout()
    save_adult_fig(fig, "adult_comparison_metrics.png")


# ---------------------------------------------------------------------------
# 6. LAPORAN & TABEL PERBANDINGAN
# ---------------------------------------------------------------------------
def _fmt(value, spec):
    """Format angka; kalau bukan angka (mis. 'N/A') tampilkan apa adanya."""
    return format(value, spec) if isinstance(value, (int, float)) else "N/A"


def generate_adult_analysis_report(comp, k, eps, min_samples, n_features):
    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, "analysis_report.md")

    km, fcm, db = comp["K-Means"], comp["Fuzzy C-Means"], comp["DBSCAN"]

    content = f"""# Laporan Analisis Eksperimen Dataset Adult (Census Income)

## 1. Pendahuluan
Eksperimen ini menerapkan algoritma klastering **K-Means**, **Fuzzy C-Means (FCM)**, dan **DBSCAN** pada dataset **Adult** (*Census Income*), yang dijalankan secara terpisah dari dataset *Mall Customers*. Tujuan dari eksperimen ini adalah mengevaluasi kinerja klastering pada data berdimensi tinggi dengan atribut campuran (*mixed-type attributes: numerical & categorical*).

## 2. Preprocessing Data
- **Pengambilan Sampel**: Dataset disampling secara acak sebanyak {SAMPLE_SIZE:,} baris untuk efisiensi komputasi *unsupervised clustering* dan perhitungan metrik jarak berdimensi tinggi.
- **Pembersihan Nilai Hilang**: Nilai yang hilang (`?`) diimputasi menggunakan median (fitur numerik) dan modus (fitur kategorikal).
- **Encoding & Scaling**:
  - Fitur numerik diskalakan menggunakan `StandardScaler`.
  - Fitur kategorikal dikodekan menggunakan `OneHotEncoder` (menghasilkan matriks fitur berdimensi {n_features}).
- **Pemisahan Target**: Kolom target pendapatan (`income`) dikeluarkan secara ketat dari input klastering *unsupervised*.

## 3. Hasil Eksperimen dan Tabel Perbandingan

| Metrik Evaluasi | K-Means ($k={k}$) | Fuzzy C-Means ($c={k}$) | DBSCAN ($\\text{{eps}}={eps}, \\text{{min}}={min_samples}$) |
|---|---:|---:|---:|
| **Jumlah Klaster** | {km['n_clusters']} | {fcm['n_clusters']} | {db['n_clusters']} |
| **Silhouette Score** | {_fmt(km['silhouette_score'], '.4f')} | {_fmt(fcm['silhouette_score'], '.4f')} | {_fmt(db['silhouette_score'], '.4f')} |
| **Davies-Bouldin Index** | {_fmt(km['davies_bouldin_index'], '.4f')} | {_fmt(fcm['davies_bouldin_index'], '.4f')} | {_fmt(db['davies_bouldin_index'], '.4f')} |
| **Calinski-Harabasz Index** | {_fmt(km['calinski_harabasz_index'], '.2f')} | {_fmt(fcm['calinski_harabasz_index'], '.2f')} | {_fmt(db['calinski_harabasz_index'], '.2f')} |
| **Runtime (detik)** | {km['runtime']:.4f} | {fcm['runtime']:.4f} | {db['runtime']:.4f} |
| **Noise Percentage** | N/A | N/A | {db['noise_percentage']:.2f}% |
| **FPC (Fuzzy)** | N/A | {fcm.get('fpc', float('nan')):.4f} | N/A |

## 4. Pembahasan dan Keterbatasan Metrik
1. **Dimensi Tinggi**: Ruang fitur *one-hot encoded* dataset Adult memiliki dimensi besar ({n_features} fitur), yang menyebabkan fenomena *curse of dimensionality* pada metrik jarak seperti *Silhouette Score*.
2. **K-Means vs Fuzzy C-Means**: K-Means dan FCM menghasilkan partisi yang mirip karena sama-sama berbasis inisialisasi centroid pada ruang dimensi tinggi. FCM memberikan keunggulan tambahan berupa derajat keanggotaan (*membership matrix*) dan FPC.
3. **DBSCAN**: Pada ruang berdimensi tinggi dengan densitas kontinu, DBSCAN memerlukan penyetelan $\\varepsilon$ yang sangat hati-hati agar tidak mengkategorikan sebagian besar data sebagai *noise*.

## 5. Kesimpulan
Eksperimen dataset Adult membuktikan bahwa pipeline *preprocessing* data campuran dan algoritma klastering berhasil diimplementasikan secara modular dan terisolasi, memberikan wawasan mendalam mengenai tantangan *unsupervised learning* pada data demografi berdimensi tinggi.
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Adult analysis report generated at {report_path}")


def print_comparison_table(metrics_path):
    """Cetak tabel perbandingan ke terminal dari file metrics JSON."""
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    header = (f"{'Model':<20} | {'Clusters':<10} | {'Silhouette':<12} | {'DBI':<10} | "
              f"{'CH Index':<10} | {'Noise (%)':<10} | {'Runtime (s)':<12}")
    line = "=" * len(header)
    print(line)
    print("PERBANDINGAN KINERJA CLUSTERING (ADULT / CENSUS INCOME)")
    print(line)
    print(header)
    print("-" * len(header))
    for name, m in metrics.items():
        print(f"{name:<20} | {str(m.get('n_clusters', 'N/A')):<10} | "
              f"{_fmt(m.get('silhouette_score'), '.4f'):<12} | "
              f"{_fmt(m.get('davies_bouldin_index'), '.4f'):<10} | "
              f"{_fmt(m.get('calinski_harabasz_index'), '.4f'):<10} | "
              f"{_fmt(m.get('noise_percentage'), '.2f'):<10} | "
              f"{_fmt(m.get('runtime'), '.4f'):<12}")
    print(line)


# ---------------------------------------------------------------------------
# 7. MAIN: JALANKAN SELURUH EKSPERIMEN
# ---------------------------------------------------------------------------
def main():
    print("=== Starting Adult Dataset Clustering Experiment ===")

    # 1. Load data (sampling 5000 baris)
    df = load_adult_data()

    # 2. Preprocessing
    X_df, X_scaled, preprocessor, y_true = preprocess_adult_data(df)
    print(f"Preprocessed Adult feature matrix shape: {X_scaled.shape}")

    # 3. K-Means
    kmeans_res = run_adult_kmeans(X_scaled, n_clusters=N_CLUSTERS)
    kmeans_metrics = evaluate_adult_clustering(X_scaled, kmeans_res["labels"])
    kmeans_metrics["runtime"] = kmeans_res["runtime"]
    kmeans_metrics["noise_percentage"] = 0.0
    print(f"K-Means (k={N_CLUSTERS}) - Silhouette: {kmeans_metrics.get('silhouette_score')}, "
          f"DBI: {kmeans_metrics.get('davies_bouldin_index')}, "
          f"CH: {kmeans_metrics.get('calinski_harabasz_index')}")
    plot_adult_pca_scatter(X_scaled, kmeans_res["labels"],
                           "K-Means Clustering (k=2) - Adult PCA", "adult_kmeans_clusters.png")

    # 4. Fuzzy C-Means
    fcm_res = run_adult_fuzzy_cmeans(X_scaled, n_clusters=N_CLUSTERS, m=FCM_M)
    fcm_metrics = evaluate_adult_clustering(X_scaled, fcm_res["labels"])
    fcm_metrics["runtime"] = fcm_res["runtime"]
    fcm_metrics["noise_percentage"] = 0.0
    fcm_metrics["fpc"] = float(fcm_res["fpc"])
    print(f"Fuzzy C-Means (c={N_CLUSTERS}, m={FCM_M}) - "
          f"Silhouette: {fcm_metrics.get('silhouette_score')}, FPC: {fcm_res['fpc']:.4f}")
    plot_adult_pca_scatter(X_scaled, fcm_res["labels"],
                           "Fuzzy C-Means Clustering (c=2, m=2.0) - Adult PCA",
                           "adult_fuzzy_cmeans_clusters.png")

    # 5. DBSCAN
    dbscan_res = run_adult_dbscan(X_scaled, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
    dbscan_metrics = evaluate_adult_clustering(X_scaled, dbscan_res["labels"], is_dbscan=True)
    dbscan_metrics["runtime"] = dbscan_res["runtime"]
    dbscan_metrics["n_clusters"] = dbscan_res["n_clusters"]
    dbscan_metrics["n_noise"] = dbscan_res["n_noise"]
    dbscan_metrics["noise_percentage"] = dbscan_res["noise_percentage"]
    print(f"DBSCAN (eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES}) - "
          f"Clusters: {dbscan_res['n_clusters']}, Noise: {dbscan_res['n_noise']} "
          f"({dbscan_res['noise_percentage']:.2f}%), "
          f"Silhouette: {dbscan_metrics.get('silhouette_score')}")
    plot_adult_pca_scatter(X_scaled, dbscan_res["labels"],
                           f"DBSCAN Clustering (eps={DBSCAN_EPS}, min={DBSCAN_MIN_SAMPLES}) - Adult PCA",
                           "adult_dbscan_clusters.png", is_dbscan=True)

    # 6. Perbandingan & simpan metrics
    comparison_data = {
        "K-Means": kmeans_metrics,
        "Fuzzy C-Means": fcm_metrics,
        "DBSCAN": dbscan_metrics,
    }
    plot_adult_comparison(comparison_data)

    os.makedirs(METRICS_DIR, exist_ok=True)
    metrics_path = os.path.join(METRICS_DIR, "adult_experiment_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(comparison_data, f, indent=4)
    print(f"Adult metrics saved to {metrics_path}")

    # 7. Laporan analisis + tabel perbandingan di terminal
    generate_adult_analysis_report(comparison_data, N_CLUSTERS, DBSCAN_EPS,
                                   DBSCAN_MIN_SAMPLES, X_scaled.shape[1])
    print_comparison_table(metrics_path)
    print("=== Adult Experiment Completed Successfully ===")


if __name__ == "__main__":
    main()
