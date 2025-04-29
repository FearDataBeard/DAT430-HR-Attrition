#project 1
# Libraries
import pandas as pd
import numpy as np
import glob
import os

# Set folder path
folder_path = r"C:\Users\Rand\Downloads\DAT-430\DAT-430"

# Step 1: Load all HRData(1–14) files
hr_files = glob.glob(os.path.join(folder_path, "HRData*.csv"))  # Adjust pattern if needed
hr_data_combined = pd.concat([pd.read_csv(f) for f in hr_files], ignore_index=True)

# Step 2: Load HR Training Data
hr_training = pd.read_csv(os.path.join(folder_path, "HR Training Data.csv"))

# Step 3: Drop duplicates before merging
hr_data_combined.drop_duplicates(inplace=True)
hr_training.drop_duplicates(inplace=True)

# Step 4: Merge on 'EmployeeNumber'
merged_df = pd.merge(hr_data_combined, hr_training, on='EmployeeNumber', how='left')

# Step 5: Drop duplicates after merging
merged_df.drop_duplicates(inplace=True)

# Step 6: Handle suffixes from merge (_x, _y)
duplicate_cols = [col.replace('_x', '') for col in merged_df.columns 
                  if '_x' in col and col.replace('_x', '') + '_y' in merged_df.columns]

# Drop _y columns
merged_df.drop(columns=[col + '_y' for col in duplicate_cols], inplace=True)

# Rename _x columns back to original
merged_df.rename(columns={col + '_x': col for col in duplicate_cols}, inplace=True)

# Step 7: Convert object columns to numeric where needed
merged_df['MonthlyIncome'] = pd.to_numeric(merged_df['MonthlyIncome'], errors='coerce')
merged_df['EmployeeNumber'] = pd.to_numeric(merged_df['EmployeeNumber'], errors='coerce')

# Step 8: Handle missing values
# Fill 'training' with 0 if missing
if 'training' in merged_df.columns:
    merged_df['training'] = merged_df['training'].fillna(0)

# Fill numeric columns with median
numeric_cols = merged_df.select_dtypes(include=['number']).columns
merged_df[numeric_cols] = merged_df[numeric_cols].fillna(merged_df[numeric_cols].median())

# Step 9: Drop constant columns (e.g., Over18, StandardHours if constant)
for col in merged_df.columns:
    if merged_df[col].nunique() <= 1:
        merged_df.drop(columns=col, inplace=True)

# ✅ Step 10: Convert Attrition to numeric (Yes=1, No=0), then fill NaN with 0 (assume no attrition)
if 'Attrition' in merged_df.columns:
    merged_df['Attrition'] = merged_df['Attrition'].map({'Yes': 1, 'No': 0})
    merged_df['Attrition'] = merged_df['Attrition'].fillna(0).astype(int)

# ✅ Step 11: Save to Excel
output_path = os.path.join(folder_path, "HRData_Combined_Cleaned.xlsx")
merged_df.to_excel(output_path, index=False)

# ✅ Step 12: Confirm
print("✅ Final cleaned dataset saved to Excel.")
print("🧾 File Path:", output_path)
print("📐 Final Shape:", merged_df.shape)
print("📊 Attrition Distribution:\n", merged_df['Attrition'].value_counts())
print("🧼 Missing Values:\n", merged_df.isnull().sum())

#project 2
# =======================
# 1. IMPORT LIBRARIES
# =======================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import shap

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
    roc_auc_score
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings("ignore")


# =========================================
# 2. LOAD & PREPARE THE DATA
# =========================================
df = pd.read_excel(r"C:\Users\Rand\Downloads\DAT-430\DAT-430\HRData_Combined_Cleaned.xlsx")

if "OverTime" in df.columns:
    df["OverTime_Encoded"] = df["OverTime"].map({"Yes": 1, "No": 0})
else:
    raise KeyError("The column 'OverTime' is missing from the dataset.")

# Create JobRole_Grouped immediately after loading
if "JobRole" in df.columns:
    jobrole_counts = df["JobRole"].value_counts()
    df["JobRole_Grouped"] = df["JobRole"].apply(
        lambda x: x if jobrole_counts[x] >= 5 else "Other"
    )
else:
    raise KeyError("The column 'JobRole' is missing from the dataset.")


print("Unique values in Attrition:", df["Attrition"].unique())
print(f"Baseline attrition: {df['Attrition'].mean()*100:.2f}%")


# =========================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# =========================================
df.describe(include='all')

plt.figure(figsize=(10,6))
sns.boxplot(x="Attrition", y="MonthlyIncome", data=df)
plt.title("Monthly Income vs Attrition")
plt.show()

plt.figure(figsize=(12,6))
sns.countplot(x="Department", hue="Attrition", data=df, palette=["blue","red"])
plt.title("Attrition Rate by Department")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12,6))
sns.boxplot(x="JobRole_Grouped", y="MonthlyIncome", hue="Attrition", data=df, palette=["blue","red"])
plt.title("Monthly Income by Job Role")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12,6))
sns.boxplot(x="JobRole_Grouped", y="MonthlyIncome", hue="JobSatisfaction", data=df, palette="coolwarm")
plt.title("Income vs Job Role & Satisfaction")
plt.xticks(rotation=45)
plt.show()


sns.set(font_scale=1.1)  

plt.figure(figsize=(16, 12))                      # wider/taller canvas
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True,                                  # force square cells
    linewidths=0.5,                               # subtle cell borders
    cbar_kws={"shrink":0.75},                     # shrink colorbar
    annot_kws={"size":10}                         # per‐cell font size
)
plt.title("Correlation Heatmap of Numeric Features", fontsize=18)
plt.xticks(rotation=75, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


g = sns.FacetGrid(df, col="Attrition", hue="JobRole_Grouped", height=6)
g.map(sns.boxplot, "JobRole_Grouped", "MonthlyIncome",
      order=sorted(df["JobRole_Grouped"].unique()))
g.set_xticklabels(rotation=45)
plt.subplots_adjust(top=0.9)
g.figure.suptitle("Income Trends by Job Role & Attrition")
plt.show()


# =========================================
# 4. FEATURE ENGINEERING & MODELING
# =========================================
df = pd.get_dummies(df, columns=['BusinessTravel','Department','JobRole'], drop_first=True)

features = ["Age","MonthlyIncome","OverTime_Encoded","JobSatisfaction"] + \
           [c for c in df.columns if c.startswith(("BusinessTravel_","Department_","JobRole_"))]
X = df[features]
y = df["Attrition"]

# 5. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 6. IMPUTE & ENCODE
num_cols = X_train.select_dtypes(include=['int64','float64']).columns
cat_cols = X_train.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer([
    ('num', SimpleImputer(strategy="median"), num_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy="most_frequent")),
        ('enc', OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols)
])

pipeline = Pipeline([('prep', preprocessor)])
X_train_imputed_scaled = pipeline.fit_transform(X_train)
X_test_imputed_scaled  = pipeline.transform(X_test)

# Extract true feature names after preprocessing
num_features = list(num_cols)
cat_features = pipeline.named_steps['prep'] \
    .named_transformers_['cat'] \
    .named_steps['enc'] \
    .get_feature_names_out(cat_cols).tolist()
all_feature_names = num_features + cat_features
print(f"Features after preprocessing: {len(all_feature_names)}")

# 7. HANDLE CLASS IMBALANCE (SMOTE)
X_train_balanced, y_train_balanced = SMOTE(random_state=42) \
    .fit_resample(X_train_imputed_scaled, y_train)

# 8. XGBOOST BENCHMARK
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
xgb_clf.fit(X_train_balanced, y_train_balanced)
pred_xgb = xgb_clf.predict(X_test_imputed_scaled)
print("XGBoost Recall:", recall_score(y_test, pred_xgb))
print(classification_report(y_test, pred_xgb))

# 9. HYPERPARAMETER TUNING (RF)
param_grid = {
    'n_estimators': [100,200,300],
    'max_depth': [None,10,20,30],
    'min_samples_split': [2,5,10],
    'min_samples_leaf': [1,2,4]
}
rf_gs = GridSearchCV(RandomForestClassifier(random_state=42),
                     param_grid, cv=5, n_jobs=-1)
rf_gs.fit(X_train_balanced, y_train_balanced)

print("Best RF params:", rf_gs.best_params_)
best_rf = rf_gs.best_estimator_

# === Experiment: Train RF on Top 3 Features Only ===
# 1. Identify indices of the top 3 importances
top3_idx = np.argsort(best_rf.feature_importances_)[::-1][:3]
top3_names = [all_feature_names[i] for i in top3_idx]
print("Top 3 features:", top3_names)

# 2. Subset the preprocessed train/test matrices
X_tr_top3 = X_train_imputed_scaled[:, top3_idx]
X_te_top3 = X_test_imputed_scaled[:, top3_idx]

# 3. Retrain a fresh RF on only those 3 columns
rf3 = RandomForestClassifier(**rf_gs.best_params_, random_state=42)
rf3.fit(X_train_balanced[:, top3_idx], y_train_balanced)

# 4. Evaluate on the test slice
pred3 = rf3.predict(X_te_top3)
print("RF (Top 3) Classification Report:")
print(classification_report(y_test, pred3))

# 10. TRAIN & PREDICT (RF)
best_rf.fit(X_train_balanced, y_train_balanced)
pred_rf = best_rf.predict(X_test_imputed_scaled)

# 11. THRESHOLD SWEEP
probs = best_rf.predict_proba(X_test_imputed_scaled)[:,1]
for t in [0.3,0.4,0.5,0.6]:
    print(f"Threshold {t:.2f} → Recall: {recall_score(y_test, (probs>t).astype(int)):.3f}")

# 12. RANDOM FOREST EVALUATION
print("\nRandom Forest classification report")
print(classification_report(y_test, pred_rf))
ConfusionMatrixDisplay.from_predictions(y_test, pred_rf, display_labels=["Stayed","Left"])
plt.show()

# 13. RANDOM FOREST FEATURE IMPORTANCES
importances = best_rf.feature_importances_
idx = np.argsort(importances)[::-1][:10]
top_feats = np.array(all_feature_names)[idx]
top_imp   = importances[idx]
threshold = 0.20
colors    = ['red' if v>threshold else 'blue' for v in top_imp]

plt.figure(figsize=(10,6))
plt.bar(top_feats, top_imp, color=colors)
plt.xticks(rotation=45, ha='right')
plt.title("Top 10 Feature Importances")
plt.show()

# 14. VOTING CLASSIFIER
logreg = LogisticRegression(class_weight='balanced', max_iter=1000)
ensemble = VotingClassifier(
    estimators=[('lr', logreg), ('rf', best_rf), ('xgb', xgb_clf)],
    voting='soft'
)
ensemble.fit(X_train_balanced, y_train_balanced)
pred_ens = ensemble.predict(X_test_imputed_scaled)
print("Ensemble Recall:", recall_score(y_test, pred_ens))
print(classification_report(y_test, pred_ens))

# ──────────────────────────────────────────────────────────────
# 9.1 K‑FOLD CROSS‑VALIDATION (5‑fold) FOR RANDOM FOREST
# ──────────────────────────────────────────────────────────────
from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.pipeline import Pipeline as ImbPipeline

# Build an imbalanced‑aware pipeline:
pipeline_kfold = ImbPipeline([
    ('prep',     preprocessor),               # ColumnTransformer
    ('smote',    SMOTE(random_state=42)),     # balance each fold
    ('rf',       RandomForestClassifier(**rf_gs.best_params_, random_state=42))
])

# Define the 5‑fold splitter:
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Run cross_val_score:
recall_scores = cross_val_score(
    pipeline_kfold,
    X,                 # full feature set before any train/test split
    y,
    cv=cv,
    scoring='recall',
    n_jobs=-1
)

print("5‑Fold Recall scores:", np.round(recall_scores, 3))
print(f"Mean Recall: {recall_scores.mean():.3f} ± {recall_scores.std():.3f}")

# =================================================
# 15. SHAP ANALYSIS & TIMING (bar‑only, robust)
#Select between full RF/ top 5 RF/ top 3 RF for SHAP analysis
#depending on available memory and time
# =================================================

import time

shap_times = {}

# 15.1 — Full RF explainer & values
explainer_full = shap.TreeExplainer(best_rf)
sv_full        = explainer_full.shap_values(X_test_imputed_scaled)
shap_vals_full = sv_full[1]  # “Left” class
n_full         = shap_vals_full.shape[1]

# 15.2 — Top 10 mean(|SHAP|) bar chart
start = time.time()
mean_abs_full = np.abs(shap_vals_full).mean(axis=0)
idx10         = np.argsort(mean_abs_full)[::-1][:min(10, n_full)]
feat10        = [all_feature_names[i] for i in idx10]
vals10        = mean_abs_full[idx10]

plt.figure(figsize=(10, 5))
plt.bar(feat10, vals10, color='steelblue')
plt.xticks(rotation=45, ha='right')
plt.title("Top 10 mean |SHAP| Values (Full RF)")
plt.tight_layout()
plt.show()
shap_times['top10_bar_s'] = time.time() - start

# 15.3 — Top 5 mean(|SHAP|) bar chart
start = time.time()
n5   = min(5, n_full)
idx5 = idx10[:n5]
feat5 = [all_feature_names[i] for i in idx5]
vals5 = mean_abs_full[idx5]

plt.figure(figsize=(8, 4))
plt.bar(feat5, vals5, color='cornflowerblue')
plt.xticks(rotation=45, ha='right')
plt.title("Top 5 mean |SHAP| Values")
plt.tight_layout()
plt.show()
shap_times['top5_bar_s'] = time.time() - start

# 15.4 — Top 3 mean(|SHAP|) bar chart on slim rf3 model
start = time.time()
sv3           = shap.TreeExplainer(rf3).shap_values(X_te_top3)
shap_vals3    = sv3[1]  # “Left” class
mean_abs_3    = np.abs(shap_vals3).mean(axis=0)
n3            = mean_abs_3.shape[0]       
slim_names    = top3_names[:n3]
slim_vals     = mean_abs_3[:n3]

plt.figure(figsize=(6, 4))
plt.bar(slim_names, slim_vals, color='seagreen')
plt.xticks(rotation=45, ha='right')
plt.title("Top 3 mean |SHAP| Values (Slim RF3)")
plt.tight_layout()
plt.show()
shap_times['top3_bar_s'] = time.time() - start

# 15.5 — Print all SHAP timings
timings = pd.Series(shap_times, name='seconds')
print("\nSHAP timings (seconds):")
print(timings)