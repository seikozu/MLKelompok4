import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

RANDOM_STATE = 42
os.makedirs('hasil_orang1_final', exist_ok=True)

# ------------------------------------------------------
# 1. Load dataset (data sekunder: California Housing Prices)
# ------------------------------------------------------
data = pd.read_csv('housing.csv')

print(data.head())
print(f"\nUkuran data: {data.shape}")

# ------------------------------------------------------
# 2. EDA singkat
# ------------------------------------------------------
print("\n=== Info & Statistik Deskriptif ===")
data.info()
print(data.describe().T)

print("\nMissing value per kolom:")
print(data.isnull().sum())

print("\nDistribusi ocean_proximity:")
print(data['ocean_proximity'].value_counts())

korelasi = data.select_dtypes(include=np.number).corr()
print("\nKorelasi terhadap median_house_value:")
print(korelasi['median_house_value'].drop('median_house_value').sort_values(ascending=False))

# Histogram semua fitur numerik
data.hist(bins=40, figsize=(12, 8), color='steelblue')
plt.suptitle("Distribusi Fitur Numerik")
plt.tight_layout()
plt.show()

# Heatmap korelasi
plt.figure(figsize=(9, 7))
plt.imshow(korelasi, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(label='Korelasi')
plt.xticks(range(len(korelasi)), korelasi.columns, rotation=60, ha='right')
plt.yticks(range(len(korelasi)), korelasi.columns)
plt.title("Heatmap Korelasi")
plt.tight_layout()
plt.show()

# Sebaran geografis (warna = harga rumah)
plt.figure(figsize=(8, 7))
sc = plt.scatter(data['longitude'], data['latitude'], c=data['median_house_value'],
                 cmap='viridis', s=6, alpha=0.5)
plt.colorbar(sc, label='median_house_value')
plt.title("Sebaran Geografis Harga Rumah")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.show()

# ------------------------------------------------------
# 3. Cleaning data (hapus duplikat)
# Missing value ditangani pada tahap preprocessing setelah split
# ------------------------------------------------------
print(f"\nJumlah data duplikat: {data.duplicated().sum()}")
data = data.drop_duplicates().reset_index(drop=True)

# ------------------------------------------------------
# 4. Persiapan fitur dan target
# ------------------------------------------------------
X = data.drop(columns='median_house_value')
y = data['median_house_value']

fitur_kategorik = ['ocean_proximity']
fitur_numerik = X.select_dtypes(include=np.number).columns.tolist()

# ------------------------------------------------------
# 5. Split data (80% training, 20% testing)
# ------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

print(f"Data train: {len(X_train)} baris | Data test: {len(X_test)} baris")

# Simpan hasil split agar digunakan Orang 2 dan Orang 3
X_train.to_csv('hasil_orang1_final/X_train.csv')
X_test.to_csv('hasil_orang1_final/X_test.csv')
y_train.to_csv('hasil_orang1_final/y_train.csv')
y_test.to_csv('hasil_orang1_final/y_test.csv')

with open('hasil_orang1_final/split_info.json', 'w') as f:
    json.dump({
        'random_state': RANDOM_STATE,
        'test_size': 0.2,
        'baris_train': len(X_train),
        'baris_test': len(X_test),
        'kolom_fitur': list(X_train.columns)
    }, f, indent=2)

# ------------------------------------------------------
# 6. Preprocessing: imputasi, encoding, dan scaling
# ------------------------------------------------------
preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ]), fitur_numerik),

    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
     fitur_kategorik),
])

# Simpan preprocessor agar Orang 2 dan 3 memakai preprocessing yang sama
joblib.dump(preprocessor, 'hasil_orang1_final/preprocessor.joblib')

# Cek hasil preprocessing pada data train
X_train_prep = preprocessor.fit_transform(X_train)

print(f"\nBentuk data train sebelum preprocessing: {X_train.shape}")
print(f"Bentuk data train setelah preprocessing: {X_train_prep.shape}")
print(f"Sisa missing value setelah imputasi    : {np.isnan(X_train_prep).sum()}")

# ------------------------------------------------------
# 7. Model AdaBoost + tuning GridSearchCV
# ------------------------------------------------------
pipe = Pipeline([
    ('prep', preprocessor),
    ('model', AdaBoostRegressor(
        estimator=DecisionTreeRegressor(random_state=RANDOM_STATE),
        random_state=RANDOM_STATE
    )),
])

param_grid = {
    'model__n_estimators': [50, 100, 150],
    'model__learning_rate': [0.1, 0.5, 1.0],
    'model__estimator': [
        DecisionTreeRegressor(max_depth=4, random_state=RANDOM_STATE),
        DecisionTreeRegressor(max_depth=6, random_state=RANDOM_STATE),
        DecisionTreeRegressor(max_depth=8, random_state=RANDOM_STATE),
    ],
}

grid = GridSearchCV(
    pipe, param_grid,
    cv=KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

model = grid.best_estimator_

print("\nParameter terbaik AdaBoost:")
for k, v in grid.best_params_.items():
    print(f"  {k}: {v}")

# ------------------------------------------------------
# 8. Simpan model AdaBoost untuk Orang 2
# ------------------------------------------------------
joblib.dump(model, 'hasil_orang1_final/adaboost_model.joblib')

print("\nModel disimpan di 'hasil_orang1_final/adaboost_model.joblib'")
