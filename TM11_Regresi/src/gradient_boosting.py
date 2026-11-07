import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

# 1. SETUP & DATA LOADING
print("--- LOADING DATA ---")
BASE_DIR = "hasil_orang1_final"
X_train = pd.read_csv(os.path.join(BASE_DIR, "X_train.csv"), index_col=0)
X_test = pd.read_csv(os.path.join(BASE_DIR, "X_test.csv"), index_col=0)
y_train = pd.read_csv(os.path.join(BASE_DIR, "y_train.csv"), index_col=0).squeeze("columns")
y_test = pd.read_csv(os.path.join(BASE_DIR, "y_test.csv"), index_col=0).squeeze("columns")

# Ensure directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Re-create preprocessor to avoid version incompatibility (same logic as Orang 1)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

num_features = ["longitude", "latitude", "housing_median_age", "total_rooms", 
                "total_bedrooms", "population", "households", "median_income"]
cat_features = ["ocean_proximity"]

num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", num_transformer, num_features),
    ("cat", cat_transformer, cat_features)
])

# Load existing AdaBoost model
adaboost_model = joblib.load(os.path.join(BASE_DIR, "adaboost_model.joblib"))

# 2. GRADIENT BOOSTING PIPELINE
print("\n--- INITIALIZING GRADIENT BOOSTING PIPELINE ---")
gb_pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", GradientBoostingRegressor(random_state=42))
])

# 3. HYPERPARAMETER TUNING
print("\n--- HYPERPARAMETER TUNING (RANDOMIZED SEARCH) ---")
param_distributions = {
    "model__n_estimators": [50, 100, 150, 200, 250, 300],
    "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.15, 0.2],
    "model__max_depth": [2, 3, 4, 5],
    "model__subsample": [0.6, 0.8, 1.0]
}

cv = KFold(n_splits=3, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(
    estimator=gb_pipeline,
    param_distributions=param_distributions,
    n_iter=40,
    cv=cv,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    random_state=42,
    verbose=1
)

start_time = time.time()
random_search.fit(X_train, y_train)
end_time = time.time()
tuning_time = end_time - start_time

best_gb_model = random_search.best_estimator_
print(f"Best Parameters: {random_search.best_params_}")
print(f"Best Score (RMSE): {-random_search.best_score_:.2f}")
print(f"Tuning Time: {tuning_time:.2f} seconds")

# 4. EVALUASI
def evaluate_regression(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)
    
    return {
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE (%)": mape,
        "R2": r2
    }

print("\n--- EVALUATING MODELS ---")
# Gradient Boosting Predictions
y_pred_gb = best_gb_model.predict(X_test)
gb_metrics = evaluate_regression(y_test, y_pred_gb, "Gradient Boosting")

# AdaBoost Predictions
y_pred_ada = adaboost_model.predict(X_test)
ada_metrics = evaluate_regression(y_test, y_pred_ada, "AdaBoost")

# 5. MODEL COMPARISON
comparison_df = pd.DataFrame([ada_metrics, gb_metrics])
print("\nModel Comparison Table:")
print(comparison_df)

comparison_df.to_csv("outputs/model_comparison.csv", index=False)

# 6. DETERMINING BEST MODEL
best_model_row = comparison_df.loc[comparison_df['RMSE'].idxmin()]
best_model_name = best_model_row['Model']
print(f"\nBest Model: {best_model_name} (based on RMSE)")

if best_model_name == "Gradient Boosting":
    best_model = best_gb_model
else:
    best_model = adaboost_model

# 7. SAVE BEST MODEL & METADATA
print("\n--- SAVING BEST MODEL ---")
os.makedirs("saved_models", exist_ok=True)
joblib.dump(best_gb_model, "saved_models/best_model.joblib")

with open("hasil_orang1_final/split_info.json", "r") as f:
    split_info = json.load(f)

metadata = {
    "best_model_name": best_model_name,
    "target": "median_house_value",
    "random_state": 42,
    "test_size": 0.2,
    "feature_columns": split_info["kolom_fitur"],
    "evaluation_metrics": {
        "Gradient Boosting": gb_metrics,
        "AdaBoost": ada_metrics
    },
    "best_parameters_gb": random_search.best_params_,
    "tuning_time_gb": tuning_time
}

with open("models/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

# 8. VISUALIZATION
print("\n--- GENERATING VISUALIZATIONS ---")
plt.style.use('seaborn-v0_8')

# A. Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_gb, alpha=0.5, label='Gradient Boosting', color='blue')
plt.scatter(y_test, y_pred_ada, alpha=0.3, label='AdaBoost', color='orange')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted House Values')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/actual_vs_predicted.png')

# B. Model Performance Comparison
metrics_to_plot = ['MAE', 'RMSE', 'MAPE (%)', 'R2']
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

for i, metric in enumerate(metrics_to_plot):
    sns.barplot(x='Model', y=metric, data=comparison_df, ax=axes[i])
    axes[i].set_title(f'Comparison of {metric}')

plt.tight_layout()
plt.savefig('outputs/model_comparison.png')

# C. Residual Analysis
plt.figure(figsize=(12, 6))
residual_gb = y_test - y_pred_gb
residual_ada = y_test - y_pred_ada

plt.subplot(1, 2, 1)
sns.histplot(residual_gb, kde=True, color='blue', label='Gradient Boosting')
sns.histplot(residual_ada, kde=True, color='orange', label='AdaBoost')
plt.title('Residual Distribution')
plt.xlabel('Residual')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_pred_gb, residual_gb, alpha=0.5, color='blue', label='Gradient Boosting')
plt.scatter(y_pred_ada, residual_ada, alpha=0.3, color='orange', label='AdaBoost')
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residual vs Predicted')
plt.xlabel('Predicted')
plt.ylabel('Residual')
plt.legend()

plt.tight_layout()
plt.savefig('outputs/residual_plot.png')

print("\n--- PROCESS COMPLETED ---")
