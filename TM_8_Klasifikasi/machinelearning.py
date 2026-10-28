# preprocessing.py
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from scipy.stats import ttest_ind
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# ============================================================
# 1. INPUT DAN MEMBACA DATA
# ============================================================

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Mengubah TotalCharges menjadi numerik
# Data kosong/spasi akan menjadi NaN
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Kolom numerik
kolom_numerik = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

# Kolom kategorik
kolom_kategorik = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "Churn"
]


# ============================================================
# 2a. DETEKSI DAN HANDLING MISSING VALUE
# ============================================================
# Dilakukan sebelum split karena ini pembersihan kualitas data,
# bukan proses yang mempelajari pola dari data.

print("=== DETEKSI MISSING VALUE ===")

missing_info = df.isnull().sum()
print(missing_info)

# Handling missing value kolom numerik
for kolom in kolom_numerik:

    total_data = len(df)  # Update setiap iterasi
    jumlah_missing = df[kolom].isnull().sum()
    persentase_missing = (jumlah_missing / total_data) * 100

    if jumlah_missing > 0:

        print(
            f"{kolom}: {jumlah_missing} missing "
            f"({persentase_missing:.2f}%)"
        )

        if persentase_missing <= 5:
            df = df.dropna(subset=[kolom])
            print(
                f"  -> Missing <= 5%, "
                f"baris missing pada {kolom} dihapus."
            )
        else:
            mean = df[kolom].mean()
            df[kolom] = df[kolom].fillna(mean)
            print(
                f"  -> Missing > 5%, "
                f"missing pada {kolom} diisi dengan mean: {mean}"
            )

# Handling missing value kolom kategorik
for kolom in kolom_kategorik:

    total_data = len(df)  # Update setiap iterasi
    jumlah_missing = df[kolom].isnull().sum()
    persentase_missing = (jumlah_missing / total_data) * 100

    if jumlah_missing > 0:

        print(
            f"{kolom}: {jumlah_missing} missing "
            f"({persentase_missing:.2f}%)"
        )

        if persentase_missing <= 5:
            df = df.dropna(subset=[kolom])
            print(
                f"  -> Missing <= 5%, "
                f"baris missing pada {kolom} dihapus."
            )
        else:
            modus = df[kolom].mode()[0]
            df[kolom] = df[kolom].fillna(modus)
            print(
                f"  -> Missing > 5%, "
                f"missing pada {kolom} diisi dengan modus: {modus}"
            )


# ============================================================
# 2b. DETEKSI DAN HANDLING DUPLIKASI DATA
# ============================================================

print("\n=== DETEKSI DUPLIKASI DATA ===")

jumlah_duplikat = df.duplicated().sum()
print(f"Jumlah data duplikat: {jumlah_duplikat}")

if jumlah_duplikat > 0:
    df = df.drop_duplicates(keep="first")
    print(
        f"{jumlah_duplikat} baris duplikat dihapus, "
        f"baris pertama dipertahankan."
    )
    print(f"Jumlah data setelah hapus duplikat: {len(df)} baris")
else:
    print("Tidak terdapat data duplikat.")


# ------------------------------------------------------------
# Menyimpan hasil preprocessing ke TXT
# ------------------------------------------------------------

nama_file_preprocessing = "hasil_preprocessing.txt"

with open(nama_file_preprocessing, "w", encoding="utf-8") as file:
    file.write("HASIL PREPROCESSING DATA\n")
    file.write("=" * 60 + "\n\n")
    file.write("Jumlah data setelah preprocessing:\n")
    file.write(f"{len(df)} baris\n")
    file.write(f"{len(df.columns)} kolom\n\n")
    file.write("Kolom numerik:\n")
    file.write(", ".join(kolom_numerik) + "\n\n")
    file.write("Data hasil preprocessing sebelum transformasi:\n")
    file.write("=" * 60 + "\n")
    file.write(df.to_string(index=False))

print(f"\nHasil preprocessing disimpan sebagai: {nama_file_preprocessing}")


# ============================================================
# 3. ENCODING
# ============================================================
# Label Encoding dilakukan sebelum split karena hanya memetakan
# kategori ke angka (tidak mempelajari distribusi/statistik data),
# sehingga tidak menyebabkan data leakage.

print("\n=== ENCODING ===")

# Simpan info nilai unik SEBELUM encoding (untuk dropdown di UI)
X_raw = df.drop(columns=["customerID", "Churn"])
kolom_info = {}
for col in X_raw.select_dtypes(include=["object"]).columns:
    kolom_info[col] = sorted(X_raw[col].unique().tolist())

# Memisahkan ID dan target
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"]

# Label Encoding untuk target (Churn)
label_enc_target = LabelEncoder()
y_encoded = label_enc_target.fit_transform(y)

# Label Encoding untuk fitur kategorikal
X_encoded = X.copy()
label_encoders = {}

for col in X_encoded.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col])
    label_encoders[col] = le

# Simpan encoder target juga
label_encoders["Churn"] = label_enc_target

target_names = label_enc_target.classes_  # ["No", "Yes"]

print(f"Target classes: {target_names}")
print(f"Jumlah fitur setelah encoding: {X_encoded.shape[1]}")


# ============================================================
# 4. SPLIT DATA (80% Training, 20% Testing)
# ============================================================
# PENTING: Split dilakukan SEBELUM outlier handling dan normalisasi
# agar tidak terjadi data leakage.

print("\n=== SPLIT DATA ===")

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print(f"Jumlah data Training (80%) : {len(X_train)} baris")
print(f"Jumlah data Testing  (20%) : {len(X_test)} baris")

# Simpan hasil split ke TXT
nama_file_split_data = "hasil_split_data.txt"

with open(nama_file_split_data, "w", encoding="utf-8") as file:
    file.write("HASIL SPLIT DATA\n")
    file.write("=" * 60 + "\n\n")
    file.write(f"Total data         : {len(X_encoded)} baris\n")
    file.write(f"Data Training (80%): {len(X_train)} baris\n")
    file.write(f"Data Testing  (20%): {len(X_test)} baris\n\n")
    file.write("X_train (Dataset Training Fitur):\n")
    file.write(X_train.to_string() + "\n\n")
    file.write("X_test (Dataset Testing Fitur):\n")
    file.write(X_test.to_string() + "\n")

print(f"Hasil split data disimpan sebagai: {nama_file_split_data}")


# ============================================================
# 5. DETEKSI OUTLIER DENGAN Z-SCORE DAN HANDLING DENGAN MEAN
# ============================================================
# PENTING: Z-score dihitung dari data TRAINING saja.
# Parameter yang sama dipakai untuk mendeteksi outlier di data testing.
# Ini mencegah data leakage.

print("\n=== DETEKSI OUTLIER DENGAN Z-SCORE (FIT DARI TRAINING) ===")

# Simpan parameter outlier untuk reproducibility
outlier_params = {}

for kolom in kolom_numerik:

    # Hitung mean dan std dari TRAINING saja
    train_mean = X_train[kolom].mean()
    train_std = X_train[kolom].std()

    # --- Proses TRAINING ---
    z_scores_train = (X_train[kolom] - train_mean) / train_std
    outlier_mask_train = np.abs(z_scores_train) > 3
    jumlah_outlier_train = outlier_mask_train.sum()

    # Mean pengganti dihitung dari non-outlier TRAINING saja
    if jumlah_outlier_train > 0:
        replacement_mean = X_train.loc[~outlier_mask_train, kolom].mean()
        X_train.loc[outlier_mask_train, kolom] = replacement_mean
    else:
        replacement_mean = train_mean

    print(f"{kolom} (train): {jumlah_outlier_train} outlier")
    if jumlah_outlier_train > 0:
        print(f"  -> Outlier diganti dengan mean non-outlier training: {replacement_mean:.4f}")

    # --- Proses TESTING (pakai parameter dari training) ---
    z_scores_test = (X_test[kolom] - train_mean) / train_std
    outlier_mask_test = np.abs(z_scores_test) > 3
    jumlah_outlier_test = outlier_mask_test.sum()

    if jumlah_outlier_test > 0:
        X_test.loc[outlier_mask_test, kolom] = replacement_mean

    print(f"{kolom} (test) : {jumlah_outlier_test} outlier")
    if jumlah_outlier_test > 0:
        print(f"  -> Outlier diganti dengan mean non-outlier training: {replacement_mean:.4f}")

    # Simpan parameter
    outlier_params[kolom] = {
        "train_mean": train_mean,
        "train_std": train_std,
        "replacement_mean": replacement_mean
    }


# ============================================================
# 6. TRANSFORMASI DATA DENGAN MIN-MAX NORMALIZATION
# ============================================================
# PENTING: Min dan Max dihitung dari data TRAINING saja.
# Parameter yang sama dipakai untuk transformasi data testing.
# Ini mencegah data leakage.

print("\n=== TRANSFORMASI MIN-MAX NORMALIZATION (FIT DARI TRAINING) ===")

normalisasi_params = {}

for kolom in kolom_numerik:

    # Hitung min dan max dari TRAINING saja (setelah outlier handling)
    train_min = X_train[kolom].min()
    train_max = X_train[kolom].max()

    # Simpan parameter SEBELUM transformasi
    normalisasi_params[kolom] = {
        "min": train_min,
        "max": train_max
    }

    # Normalisasi TRAINING
    X_train[kolom] = (X_train[kolom] - train_min) / (train_max - train_min)

    # Normalisasi TESTING (pakai min/max dari training)
    X_test[kolom] = (X_test[kolom] - train_min) / (train_max - train_min)

    print(
        f"{kolom}: "
        f"train_min = {train_min:.4f}, "
        f"train_max = {train_max:.4f}"
    )

print("\nData training numerik setelah normalisasi:")
print(X_train[kolom_numerik].head())

print("\nData testing numerik setelah normalisasi:")
print(X_test[kolom_numerik].head())

# Simpan hasil transformasi ke TXT
nama_file_transformasi = "hasil_transformasi.txt"

with open(nama_file_transformasi, "w", encoding="utf-8") as file:
    file.write("HASIL TRANSFORMASI DATA\n")
    file.write("=" * 60 + "\n\n")
    file.write("Metode transformasi:\n")
    file.write("Min-Max Normalization (fit dari data training)\n\n")
    file.write("Parameter normalisasi (dari training):\n")
    for kolom in kolom_numerik:
        p = normalisasi_params[kolom]
        file.write(f"  {kolom}: min = {p['min']:.4f}, max = {p['max']:.4f}\n")
    file.write(f"\nKolom yang ditransformasi:\n")
    file.write(", ".join(kolom_numerik) + "\n\n")
    file.write(f"Training: {len(X_train)} baris\n")
    file.write(f"Testing : {len(X_test)} baris\n\n")
    file.write("Data Training setelah Normalisasi:\n")
    file.write("=" * 60 + "\n")
    file.write(X_train.to_string() + "\n\n")
    file.write("Data Testing setelah Normalisasi:\n")
    file.write("=" * 60 + "\n")
    file.write(X_test.to_string())

print(f"\nHasil transformasi disimpan sebagai: {nama_file_transformasi}")


# ============================================================
# 7. SELEKSI FITUR (Chi-Square & T-Test)
# ============================================================
# Dilakukan pada data TRAINING saja — sudah benar dari awal.

print("\n=== SELEKSI FITUR ===")

nama_file_seleksi_fitur = "hasil_seleksi_fitur.txt"

# Chi-Square (Khusus Fitur Kategorikal)
kolom_kategorik_fitur = [
    k for k in kolom_kategorik
    if k not in ["customerID", "Churn"]
]

k_terbaik = 5
selector_chi2 = SelectKBest(score_func=chi2, k=k_terbaik)
selector_chi2.fit(X_train[kolom_kategorik_fitur], y_train)

fitur_terpilih_chi2 = [
    kolom_kategorik_fitur[i]
    for i in selector_chi2.get_support(indices=True)
]

print(f"Fitur kategorikal terpilih (Chi-Square Top-{k_terbaik}):")
print(fitur_terpilih_chi2)

# T-Test (Khusus Fitur Numerik Kontinu)
fitur_terpilih_ttest = []
hasil_ttest_log = []

print("\nHasil Uji T-Test pada Kolom Numerik:")
for col in kolom_numerik:
    grup_0 = X_train[y_train == 0][col]
    grup_1 = X_train[y_train == 1][col]

    stat, p_val = ttest_ind(grup_0, grup_1)

    if p_val < 0.05:
        fitur_terpilih_ttest.append(col)
        status = "Signifikan"
    else:
        status = "Tidak Signifikan"

    pesan = f"  - {col}: t-stat = {stat:.4f}, p-value = {p_val:.4e} ({status})"
    print(pesan)
    hasil_ttest_log.append(pesan)

print(f"\nFitur numerik signifikan (T-Test):")
print(fitur_terpilih_ttest)

# Gabungkan fitur terpilih (tanpa duplikat, jaga urutan)
fitur_klasifikasi = list(
    dict.fromkeys(
        fitur_terpilih_chi2 + fitur_terpilih_ttest
    )
)

print(f"\nFitur gabungan untuk klasifikasi:")
for nomor, fitur in enumerate(fitur_klasifikasi, start=1):
    print(f"{nomor}. {fitur}")
print(f"Total fitur terpilih: {len(fitur_klasifikasi)}")

# Simpan hasil seleksi fitur ke TXT
with open(nama_file_seleksi_fitur, "w", encoding="utf-8") as file:
    file.write("HASIL SELEKSI FITUR\n")
    file.write("=" * 60 + "\n\n")
    file.write(f"1. CHI-SQUARE (Top-{k_terbaik} Fitur Kategorikal):\n")
    file.write(", ".join(fitur_terpilih_chi2) + "\n\n")
    file.write("2. T-TEST (Fitur Numerik Kontinu):\n")
    for log in hasil_ttest_log:
        file.write(log + "\n")
    file.write("\nFitur Numerik Signifikan (p-value < 0.05):\n")
    file.write(", ".join(fitur_terpilih_ttest) + "\n\n")
    file.write("3. FITUR GABUNGAN UNTUK KLASIFIKASI:\n")
    for nomor, fitur in enumerate(fitur_klasifikasi, start=1):
        file.write(f"   {nomor}. {fitur}\n")

print(f"Hasil seleksi fitur disimpan sebagai: {nama_file_seleksi_fitur}")


# ============================================================
# Menyiapkan data training & testing berdasarkan fitur terpilih
# ============================================================
# Data ini dipakai oleh SEMUA model (Decision Tree & Naive Bayes)
# supaya perbandingan fair — data yang sama, split yang sama.

X_train_model = X_train[fitur_klasifikasi]
X_test_model = X_test[fitur_klasifikasi]

print(f"\nDimensi data untuk klasifikasi:")
print(f"Training : {X_train_model.shape}")
print(f"Testing  : {X_test_model.shape}")


# ============================================================
# 8. KLASIFIKASI MENGGUNAKAN DECISION TREE
# ============================================================

print("\n" + "=" * 70)
print("8. KLASIFIKASI MENGGUNAKAN DECISION TREE")
print("=" * 70)

# Inisialisasi & training
decision_tree_model = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42
)

decision_tree_model.fit(X_train_model, y_train)

print("\nDecision Tree berhasil dilatih.")
print("Criterion   : Entropy")
print("Random State: 42")

# Prediksi
y_pred_dt = decision_tree_model.predict(X_test_model)

print("\nHasil prediksi Decision Tree:")
print(y_pred_dt)


# ============================================================
# 9. EVALUASI MODEL DECISION TREE
# ============================================================

print("\n" + "=" * 70)
print("9. EVALUASI MODEL DECISION TREE")
print("=" * 70)

# Confusion Matrix
cm_dt = confusion_matrix(y_test, y_pred_dt)
print("\nConfusion Matrix:")
print(cm_dt)

# Komponen Confusion Matrix
# Urutan dari .ravel(): TN, FP, FN, TP
# Sesuai dokumentasi sklearn
tn_dt, fp_dt, fn_dt, tp_dt = cm_dt.ravel()

print("\nKomponen Confusion Matrix:")
print(f"True Negative  (TN): {tn_dt}")
print(f"False Positive (FP): {fp_dt}")
print(f"False Negative (FN): {fn_dt}")
print(f"True Positive  (TP): {tp_dt}")

# Metrik evaluasi
accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt, zero_division=0)
recall_dt = recall_score(y_test, y_pred_dt, zero_division=0)
f1_dt = f1_score(y_test, y_pred_dt, zero_division=0)

# Sensitivity & Specificity
# Sensitivity = TP / (TP + FN) → kemampuan mendeteksi kelas positif
# Specificity = TN / (TN + FP) → kemampuan mendeteksi kelas negatif
sensitivity_dt = tp_dt / (tp_dt + fn_dt) if (tp_dt + fn_dt) > 0 else 0
specificity_dt = tn_dt / (tn_dt + fp_dt) if (tn_dt + fp_dt) > 0 else 0

print("\nMetrik Evaluasi:")
print("-" * 40)
print(f"Accuracy    : {accuracy_dt:.4f}")
print(f"Precision   : {precision_dt:.4f}")
print(f"Sensitivity : {sensitivity_dt:.4f}")
print(f"F1-Score    : {f1_dt:.4f}")
print(f"Specificity : {specificity_dt:.4f}")

# Classification Report
print("\nClassification Report Decision Tree:")
print(classification_report(y_test, y_pred_dt, target_names=target_names))

# Heatmap Confusion Matrix
fig_dt, ax_dt = plt.subplots(figsize=(8, 5))
sns.heatmap(
    cm_dt, annot=True, fmt="d", cmap="Blues",
    xticklabels=target_names, yticklabels=target_names, ax=ax_dt
)
ax_dt.set_xlabel("Predicted")
ax_dt.set_ylabel("Actual")
ax_dt.set_title("Confusion Matrix - Decision Tree")
plt.tight_layout()
plt.savefig("heatmap_confusion_matrix_decision_tree.png", dpi=150)
plt.close()
print("Heatmap disimpan sebagai: heatmap_confusion_matrix_decision_tree.png")

# Simpan hasil ke TXT
nama_file_dt = "hasil_decision_tree_evaluasi.txt"
with open(nama_file_dt, "w", encoding="utf-8") as file:
    file.write("HASIL KLASIFIKASI DAN EVALUASI DECISION TREE\n")
    file.write("=" * 70 + "\n\n")
    file.write("Metode       : Decision Tree\n")
    file.write("Criterion    : Entropy\n")
    file.write("Random State : 42\n\n")
    file.write("Fitur yang digunakan:\n")
    for nomor, fitur in enumerate(fitur_klasifikasi, start=1):
        file.write(f"{nomor}. {fitur}\n")
    file.write(f"\nJumlah fitur : {len(fitur_klasifikasi)}\n")
    file.write(f"Data training: {len(X_train_model)}\n")
    file.write(f"Data testing : {len(X_test_model)}\n\n")
    file.write("Confusion Matrix:\n")
    file.write(str(cm_dt) + "\n\n")
    file.write("Komponen Confusion Matrix:\n")
    file.write(f"True Negative  (TN): {tn_dt}\n")
    file.write(f"False Positive (FP): {fp_dt}\n")
    file.write(f"False Negative (FN): {fn_dt}\n")
    file.write(f"True Positive  (TP): {tp_dt}\n\n")
    file.write("Metrik Evaluasi:\n")
    file.write(f"Accuracy    : {accuracy_dt:.4f}\n")
    file.write(f"Precision   : {precision_dt:.4f}\n")
    file.write(f"Sensitivity : {sensitivity_dt:.4f}\n")
    file.write(f"F1-Score    : {f1_dt:.4f}\n")
    file.write(f"Specificity : {specificity_dt:.4f}\n\n")
    file.write("Classification Report:\n")
    file.write(classification_report(y_test, y_pred_dt, target_names=target_names))

print(f"Hasil disimpan sebagai: {nama_file_dt}")


# ============================================================
# 10. KLASIFIKASI MENGGUNAKAN NAÏVE BAYES
# ============================================================
# Menggunakan SPLIT DATA YANG SAMA dengan Decision Tree
# (X_train_model, X_test_model, y_train, y_test).
# Memastikan perbandingan yang fair antar model.

print("\n" + "=" * 70)
print("10. KLASIFIKASI MENGGUNAKAN NAÏVE BAYES")
print("=" * 70)

print("\nFitur yang digunakan (sama dengan Decision Tree):")
for nomor, fitur in enumerate(fitur_klasifikasi, start=1):
    print(f"{nomor}. {fitur}")

print(f"\nDimensi data (sama dengan Decision Tree):")
print(f"Training : {X_train_model.shape}")
print(f"Testing  : {X_test_model.shape}")

# Inisialisasi & training
naive_bayes_model = GaussianNB()
naive_bayes_model.fit(X_train_model, y_train)

print("\nNaïve Bayes (GaussianNB) berhasil dilatih.")

# Prediksi
y_pred_nb = naive_bayes_model.predict(X_test_model)

print("\nHasil prediksi Naïve Bayes:")
print(y_pred_nb)


# ============================================================
# 11. EVALUASI MODEL NAÏVE BAYES
# ============================================================

print("\n" + "=" * 70)
print("11. EVALUASI MODEL NAÏVE BAYES")
print("=" * 70)

# Confusion Matrix
cm_nb = confusion_matrix(y_test, y_pred_nb)
print("\nConfusion Matrix:")
print(cm_nb)

# Komponen Confusion Matrix
tn_nb, fp_nb, fn_nb, tp_nb = cm_nb.ravel()

print("\nKomponen Confusion Matrix:")
print(f"True Negative  (TN): {tn_nb}")
print(f"False Positive (FP): {fp_nb}")
print(f"False Negative (FN): {fn_nb}")
print(f"True Positive  (TP): {tp_nb}")

# Metrik evaluasi
accuracy_nb = accuracy_score(y_test, y_pred_nb)
precision_nb = precision_score(y_test, y_pred_nb, zero_division=0)
recall_nb = recall_score(y_test, y_pred_nb, zero_division=0)
f1_nb = f1_score(y_test, y_pred_nb, zero_division=0)

# Sensitivity & Specificity
# Rumus BENAR (berbeda dengan PDF referensi yang salah):
# Sensitivity = TP / (TP + FN)
# Specificity = TN / (TN + FP)
sensitivity_nb = tp_nb / (tp_nb + fn_nb) if (tp_nb + fn_nb) > 0 else 0
specificity_nb = tn_nb / (tn_nb + fp_nb) if (tn_nb + fp_nb) > 0 else 0

print("\nMetrik Evaluasi:")
print("-" * 40)
print(f"Accuracy    : {accuracy_nb:.4f}")
print(f"Precision   : {precision_nb:.4f}")
print(f"Sensitivity : {sensitivity_nb:.4f}")
print(f"F1-Score    : {f1_nb:.4f}")
print(f"Specificity : {specificity_nb:.4f}")

# Classification Report
print("\nClassification Report Naïve Bayes:")
print(classification_report(y_test, y_pred_nb, target_names=target_names))

# Heatmap Confusion Matrix
fig_nb, ax_nb = plt.subplots(figsize=(8, 5))
sns.heatmap(
    cm_nb, annot=True, fmt="d", cmap="Blues",
    xticklabels=target_names, yticklabels=target_names, ax=ax_nb
)
ax_nb.set_xlabel("Predicted")
ax_nb.set_ylabel("Actual")
ax_nb.set_title("Confusion Matrix - Naïve Bayes")
plt.tight_layout()
plt.savefig("heatmap_confusion_matrix_naive_bayes.png", dpi=150)
plt.close()
print("Heatmap disimpan sebagai: heatmap_confusion_matrix_naive_bayes.png")

# Simpan hasil ke TXT
nama_file_nb = "hasil_naive_bayes_evaluasi.txt"
with open(nama_file_nb, "w", encoding="utf-8") as file:
    file.write("HASIL KLASIFIKASI DAN EVALUASI NAÏVE BAYES\n")
    file.write("=" * 70 + "\n\n")
    file.write("Metode       : Naïve Bayes (GaussianNB)\n")
    file.write("Test Size    : 20%\n")
    file.write("Random State : 42\n\n")
    file.write("Fitur yang digunakan:\n")
    for nomor, fitur in enumerate(fitur_klasifikasi, start=1):
        file.write(f"{nomor}. {fitur}\n")
    file.write(f"\nJumlah fitur : {len(fitur_klasifikasi)}\n")
    file.write(f"Data training: {len(X_train_model)}\n")
    file.write(f"Data testing : {len(X_test_model)}\n\n")
    file.write("Confusion Matrix:\n")
    file.write(str(cm_nb) + "\n\n")
    file.write("Komponen Confusion Matrix:\n")
    file.write(f"True Negative  (TN): {tn_nb}\n")
    file.write(f"False Positive (FP): {fp_nb}\n")
    file.write(f"False Negative (FN): {fn_nb}\n")
    file.write(f"True Positive  (TP): {tp_nb}\n\n")
    file.write("Metrik Evaluasi:\n")
    file.write(f"Accuracy    : {accuracy_nb:.4f}\n")
    file.write(f"Precision   : {precision_nb:.4f}\n")
    file.write(f"Sensitivity : {sensitivity_nb:.4f}\n")
    file.write(f"F1-Score    : {f1_nb:.4f}\n")
    file.write(f"Specificity : {specificity_nb:.4f}\n\n")
    file.write("Classification Report:\n")
    file.write(classification_report(y_test, y_pred_nb, target_names=target_names))

print(f"Hasil disimpan sebagai: {nama_file_nb}")

print("\n" + "=" * 70)
print("KLASIFIKASI DAN EVALUASI NAÏVE BAYES SELESAI")
print("=" * 70)


# ============================================================
# 12. PERBANDINGAN KINERJA DECISION TREE vs NAÏVE BAYES
# ============================================================

print("\n" + "=" * 70)
print("12. PERBANDINGAN KINERJA MODEL")
print("=" * 70)

print(f"\n{'Metrik':<15} {'Decision Tree':>15} {'Naïve Bayes':>15}")
print("-" * 47)
print(f"{'Accuracy':<15} {accuracy_dt:>15.4f} {accuracy_nb:>15.4f}")
print(f"{'Precision':<15} {precision_dt:>15.4f} {precision_nb:>15.4f}")
print(f"{'Sensitivity':<15} {sensitivity_dt:>15.4f} {sensitivity_nb:>15.4f}")
print(f"{'F1-Score':<15} {f1_dt:>15.4f} {f1_nb:>15.4f}")
print(f"{'Specificity':<15} {specificity_dt:>15.4f} {specificity_nb:>15.4f}")

# Tentukan model terbaik
if accuracy_dt > accuracy_nb:
    model_terbaik = "Decision Tree"
    akurasi_terbaik = accuracy_dt
elif accuracy_nb > accuracy_dt:
    model_terbaik = "Naïve Bayes"
    akurasi_terbaik = accuracy_nb
else:
    model_terbaik = "Keduanya sama"
    akurasi_terbaik = accuracy_dt

print(f"\nModel dengan akurasi tertinggi: {model_terbaik} ({akurasi_terbaik:.4f})")

# Simpan perbandingan ke TXT
nama_file_perbandingan = "hasil_perbandingan_model.txt"
with open(nama_file_perbandingan, "w", encoding="utf-8") as file:
    file.write("PERBANDINGAN KINERJA MODEL\n")
    file.write("=" * 70 + "\n\n")
    file.write("Kedua model menggunakan:\n")
    file.write(f"  - Data training & testing yang SAMA (split yang sama)\n")
    file.write(f"  - Fitur yang SAMA (hasil seleksi fitur)\n")
    file.write(f"  - Normalisasi & outlier handling dari TRAINING saja\n\n")
    file.write(f"{'Metrik':<15} {'Decision Tree':>15} {'Naïve Bayes':>15}\n")
    file.write("-" * 47 + "\n")
    file.write(f"{'Accuracy':<15} {accuracy_dt:>15.4f} {accuracy_nb:>15.4f}\n")
    file.write(f"{'Precision':<15} {precision_dt:>15.4f} {precision_nb:>15.4f}\n")
    file.write(f"{'Sensitivity':<15} {sensitivity_dt:>15.4f} {sensitivity_nb:>15.4f}\n")
    file.write(f"{'F1-Score':<15} {f1_dt:>15.4f} {f1_nb:>15.4f}\n")
    file.write(f"{'Specificity':<15} {specificity_dt:>15.4f} {specificity_nb:>15.4f}\n")
    file.write(f"\nModel dengan akurasi tertinggi: {model_terbaik} ({akurasi_terbaik:.4f})\n")
    file.write("\n\nCATATAN RUMUS:\n")
    file.write("- Sensitivity = TP / (TP + FN) = Recall\n")
    file.write("  Keduanya rumus identik. 'Recall' lazim di ML,\n")
    file.write("  'Sensitivity' lazim di Statistik/Medis.\n")
    file.write("- Specificity = TN / (TN + FP)\n")
    file.write("  Kemampuan model mendeteksi kelas negatif (Churn = No)\n")

print(f"Hasil perbandingan disimpan sebagai: {nama_file_perbandingan}")


# ============================================================
# 13. SIMPAN FILE .PKL UNTUK UI/UX
# ============================================================

print("\n" + "=" * 70)
print("13. MENYIMPAN FILE UNTUK UI/UX")
print("=" * 70)

# 13.1 Model yang sudah dilatih
joblib.dump(naive_bayes_model, "model_naive_bayes.pkl")
print("Saved: model_naive_bayes.pkl")

joblib.dump(decision_tree_model, "model_decision_tree.pkl")
print("Saved: model_decision_tree.pkl")

# 13.2 Label encoders (untuk encode input user di UI)
joblib.dump(label_encoders, "label_encoders.pkl")
print("Saved: label_encoders.pkl")

# 13.3 Parameter normalisasi (min/max dari TRAINING)
joblib.dump(normalisasi_params, "normalisasi_params.pkl")
print("Saved: normalisasi_params.pkl")

# 13.4 Parameter outlier (mean/std/replacement dari TRAINING)
joblib.dump(outlier_params, "outlier_params.pkl")
print("Saved: outlier_params.pkl")

# 13.5 Daftar fitur terpilih
joblib.dump(fitur_klasifikasi, "fitur_klasifikasi.pkl")
print("Saved: fitur_klasifikasi.pkl")

# 13.6 Info kolom (nilai unik tiap kolom kategorikal, untuk dropdown UI)
joblib.dump(kolom_info, "kolom_info.pkl")
print("Saved: kolom_info.pkl")

# 13.7 Daftar kolom numerik
joblib.dump(kolom_numerik, "kolom_numerik.pkl")
print("Saved: kolom_numerik.pkl")

# 13.8 Hasil evaluasi semua model
hasil_evaluasi = {
    "decision_tree": {
        "accuracy": accuracy_dt,
        "precision": precision_dt,
        "recall": recall_dt,
        "f1": f1_dt,
        "sensitivity": sensitivity_dt,
        "specificity": specificity_dt,
        "confusion_matrix": cm_dt.tolist()
    },
    "naive_bayes": {
        "accuracy": accuracy_nb,
        "precision": precision_nb,
        "recall": recall_nb,
        "f1": f1_nb,
        "sensitivity": sensitivity_nb,
        "specificity": specificity_nb,
        "confusion_matrix": cm_nb.tolist()
    },
    "model_terbaik": model_terbaik,
    "akurasi_terbaik": akurasi_terbaik
}

joblib.dump(hasil_evaluasi, "hasil_evaluasi.pkl")
print("Saved: hasil_evaluasi.pkl")


# ============================================================
# 14. KLASIFIKASI MENGGUNAKAN SUPPORT VECTOR MACHINE (SVM)
# ============================================================

print("\n" + "=" * 70)
print("14. KLASIFIKASI MENGGUNAKAN SUPPORT VECTOR MACHINE (SVM)")
print("=" * 70)

# 14.1 Pembuatan Model SVM
svm_model = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    random_state=42
)

print("Model SVM berhasil dibuat.")
print("Kernel      : RBF")
print("C           : 1.0")
print("Gamma       : scale")
print("Random State: 42")

# 14.2 Proses Training dan Prediksi
svm_model.fit(X_train_model, y_train)

print("\nSVM berhasil dilatih menggunakan data training.")
print(f"Jumlah data training: {len(X_train_model)}")
print(f"Jumlah data testing : {len(X_test_model)}")
print(f"Jumlah fitur        : {X_train_model.shape[1]}")

y_pred_svm = svm_model.predict(X_test_model)

print("\nHasil prediksi SVM:")
print(y_pred_svm)

# 14.3 Evaluasi Model SVM
cm_svm = confusion_matrix(y_test, y_pred_svm, labels=[0, 1])

tn_svm, fp_svm, fn_svm, tp_svm = cm_svm.ravel()

accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(
    y_test, y_pred_svm, zero_division=0
)
recall_svm = recall_score(
    y_test, y_pred_svm, zero_division=0
)
f1_svm = f1_score(
    y_test, y_pred_svm, zero_division=0
)

sensitivity_svm = (
    tp_svm / (tp_svm + fn_svm)
    if (tp_svm + fn_svm) > 0 else 0
)

specificity_svm = (
    tn_svm / (tn_svm + fp_svm)
    if (tn_svm + fp_svm) > 0 else 0
)

print("\nConfusion Matrix SVM:")
print(cm_svm)

print("\nMetrik Evaluasi SVM:")
print(f"Accuracy    : {accuracy_svm:.4f}")
print(f"Precision   : {precision_svm:.4f}")
print(f"Recall      : {recall_svm:.4f}")
print(f"Sensitivity : {sensitivity_svm:.4f}")
print(f"F1-Score    : {f1_svm:.4f}")
print(f"Specificity : {specificity_svm:.4f}")

print("\nClassification Report SVM:")
print(classification_report(
    y_test,
    y_pred_svm,
    labels=[0, 1],
    target_names=target_names,
    zero_division=0
))

# Visualisasi Confusion Matrix
fig_svm, ax_svm = plt.subplots(figsize=(8, 5))

sns.heatmap(
    cm_svm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=target_names,
    yticklabels=target_names,
    ax=ax_svm
)

ax_svm.set_xlabel("Predicted")
ax_svm.set_ylabel("Actual")
ax_svm.set_title("Confusion Matrix - SVM")

plt.tight_layout()
plt.savefig("heatmap_confusion_matrix_svm.png", dpi=150)
plt.close()

print("\nHeatmap disimpan sebagai: heatmap_confusion_matrix_svm.png")

# Simpan hasil evaluasi SVM
with open("hasil_svm_evaluasi.txt", "w", encoding="utf-8") as file:
    file.write("HASIL KLASIFIKASI DAN EVALUASI SVM\n")
    file.write("=" * 70 + "\n\n")
    file.write("Metode       : Support Vector Machine\n")
    file.write("Kernel       : RBF\n")
    file.write("C            : 1.0\n")
    file.write("Gamma        : scale\n")
    file.write("Random State : 42\n\n")

    file.write("Fitur yang digunakan:\n")
    for nomor, fitur in enumerate(fitur_klasifikasi, start=1):
        file.write(f"{nomor}. {fitur}\n")

    file.write(f"\nData training: {len(X_train_model)}\n")
    file.write(f"Data testing : {len(X_test_model)}\n\n")
    file.write("Confusion Matrix:\n")
    file.write(str(cm_svm) + "\n\n")

    file.write("Komponen Confusion Matrix:\n")
    file.write(f"True Negative  (TN): {tn_svm}\n")
    file.write(f"False Positive (FP): {fp_svm}\n")
    file.write(f"False Negative (FN): {fn_svm}\n")
    file.write(f"True Positive  (TP): {tp_svm}\n\n")

    file.write("Metrik Evaluasi:\n")
    file.write(f"Accuracy    : {accuracy_svm:.4f}\n")
    file.write(f"Precision   : {precision_svm:.4f}\n")
    file.write(f"Recall      : {recall_svm:.4f}\n")
    file.write(f"Sensitivity : {sensitivity_svm:.4f}\n")
    file.write(f"F1-Score    : {f1_svm:.4f}\n")
    file.write(f"Specificity : {specificity_svm:.4f}\n\n")

    file.write("Classification Report:\n")
    file.write(classification_report(
        y_test,
        y_pred_svm,
        labels=[0, 1],
        target_names=target_names,
        zero_division=0
    ))

# Simpan model untuk digunakan aplikasi
joblib.dump(svm_model, "model_svm.pkl")

print("Hasil evaluasi disimpan: hasil_svm_evaluasi.txt")
print("Model disimpan: model_svm.pkl")


# ============================================================
# 15. PENGGUNAAN HASIL DECISION TREE DARI UTS
# ============================================================

print("\n" + "=" * 70)
print("15. PENGGUNAAN HASIL DECISION TREE DARI UTS")
print("=" * 70)

nama_model_uts = "model_decision_tree_uts.pkl"

if __import__("os").path.exists(nama_model_uts):
    decision_tree_uts = joblib.load(nama_model_uts)

    print("Model Decision Tree dari UTS berhasil dimuat.")

    # Pastikan model UTS menggunakan fitur yang kompatibel
    jumlah_fitur_uts = getattr(
        decision_tree_uts, "n_features_in_", None
    )

    if (
        jumlah_fitur_uts is not None
        and jumlah_fitur_uts != X_test_model.shape[1]
    ):
        print(
            "Model UTS tidak dapat langsung digunakan karena "
            "jumlah fiturnya berbeda dengan data pengujian saat ini."
        )
        print(f"Fitur model UTS : {jumlah_fitur_uts}")
        print(f"Fitur saat ini  : {X_test_model.shape[1]}")
        print(
            "Sesuaikan preprocessing dan fitur dengan pipeline UTS "
            "sebelum melakukan prediksi."
        )
    else:
        y_pred_dt_uts = decision_tree_uts.predict(X_test_model)

        accuracy_dt_uts = accuracy_score(
            y_test, y_pred_dt_uts
        )
        precision_dt_uts = precision_score(
            y_test, y_pred_dt_uts, zero_division=0
        )
        recall_dt_uts = recall_score(
            y_test, y_pred_dt_uts, zero_division=0
        )
        f1_dt_uts = f1_score(
            y_test, y_pred_dt_uts, zero_division=0
        )

        print("\nEvaluasi Decision Tree UTS:")
        print(f"Accuracy  : {accuracy_dt_uts:.4f}")
        print(f"Precision : {precision_dt_uts:.4f}")
        print(f"Recall    : {recall_dt_uts:.4f}")
        print(f"F1-Score  : {f1_dt_uts:.4f}")

        print("\nClassification Report Decision Tree UTS:")
        print(classification_report(
            y_test,
            y_pred_dt_uts,
            labels=[0, 1],
            target_names=target_names,
            zero_division=0
        ))

        with open(
            "hasil_decision_tree_uts.txt",
            "w",
            encoding="utf-8"
        ) as file:
            file.write("EVALUASI MODEL DECISION TREE DARI UTS\n")
            file.write("=" * 60 + "\n\n")
            file.write(f"Accuracy  : {accuracy_dt_uts:.4f}\n")
            file.write(f"Precision : {precision_dt_uts:.4f}\n")
            file.write(f"Recall    : {recall_dt_uts:.4f}\n")
            file.write(f"F1-Score  : {f1_dt_uts:.4f}\n\n")
            file.write(classification_report(
                y_test,
                y_pred_dt_uts,
                labels=[0, 1],
                target_names=target_names,
                zero_division=0
            ))

        print("Hasil disimpan: hasil_decision_tree_uts.txt")

else:
    print(
        f"File {nama_model_uts} belum ditemukan. "
        "Simpan model Decision Tree UTS dengan nama tersebut "
        "sebelum menjalankan evaluasi."
    )


print(f"\n{'Metrik':<15} {'Decision Tree':>15} "
      f"{'Naive Bayes':>15} {'SVM':>15}")
print("-" * 64)

print(f"{'Accuracy':<15} {accuracy_dt:>15.4f} "
      f"{accuracy_nb:>15.4f} {accuracy_svm:>15.4f}")

print(f"{'Precision':<15} {precision_dt:>15.4f} "
      f"{precision_nb:>15.4f} {precision_svm:>15.4f}")

print(f"{'Recall':<15} {recall_dt:>15.4f} "
      f"{recall_nb:>15.4f} {recall_svm:>15.4f}")

print(f"{'F1-Score':<15} {f1_dt:>15.4f} "
      f"{f1_nb:>15.4f} {f1_svm:>15.4f}")

print(f"{'Specificity':<15} {specificity_dt:>15.4f} "
      f"{specificity_nb:>15.4f} {specificity_svm:>15.4f}")

daftar_akurasi = {
    "Decision Tree": accuracy_dt,
    "Naive Bayes": accuracy_nb,
    "SVM": accuracy_svm
}

model_terbaik = max(daftar_akurasi, key=daftar_akurasi.get)
akurasi_terbaik = daftar_akurasi[model_terbaik]

print(
    f"\nModel dengan akurasi tertinggi: "
    f"{model_terbaik} ({akurasi_terbaik:.4f})"
)

hasil_evaluasi["svm"] = {
    "accuracy": accuracy_svm,
    "precision": precision_svm,
    "recall": recall_svm,
    "f1": f1_svm,
    "sensitivity": sensitivity_svm,
    "specificity": specificity_svm,
    "confusion_matrix": cm_svm.tolist()
}

# Perbarui model terbaik dari ketiga model
hasil_evaluasi["model_terbaik"] = model_terbaik
hasil_evaluasi["akurasi_terbaik"] = akurasi_terbaik

joblib.dump(hasil_evaluasi, "hasil_evaluasi.pkl")
print("Hasil evaluasi tiga model disimpan: hasil_evaluasi.pkl")
