import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# Paths to your raw synthetic datasets (update if needed)
datasets = {
    "hepatitis": r"D:/sam/project/data/raw_hepatitis_dataset.csv",
    "heart": r"D:/sam/project/data/raw_heart_disease_dataset.csv",
    "diabetes": r"D:/sam/project/data/raw_diabetes_dataset.csv",
    "liver": r"D:/sam/project/data/raw_liver_disease_dataset.csv",
    "lung_cancer": r"D:/sam/project/data/raw_formatted_lung_cancer_dataset.csv"
}

def preprocess_dataset(file_path, save_name, target_col="disease_label", output_dir="output"):
    """
    Reads CSV, does feature engineering, encoding, imputation, scaling,
    RFE feature selection (70% of features, min 1) and optional PCA (<=10 components).
    Saves two CSVs: selected-features and PCA-features (both include target column).
    """
    os.makedirs(output_dir, exist_ok=True)
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return

    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found in {file_path}. Available columns: {list(df.columns)}")
        return

    # -------- Feature Engineering Examples --------
    df_fe = df.copy()

    # Age groups
    if "Age" in df_fe.columns:
        # create bins - adjust as necessary
        df_fe["Age_Group"] = pd.cut(df_fe["Age"], bins=[-1, 30, 50, 150], labels=["Young", "Middle", "Old"])
        df_fe["Age_Group"] = df_fe["Age_Group"].astype(str)

    # Cholesterol / RestingBP ratio
    if {"Cholesterol", "RestingBP"}.issubset(df_fe.columns):
        # add small epsilon to avoid division by zero
        df_fe["Chol_BP_Ratio"] = df_fe["Cholesterol"] / (df_fe["RestingBP"].astype(float) + 1e-6)

    # BMI calculation if Weight (kg) and Height (cm)
    if {"Weight", "Height"}.issubset(df_fe.columns):
        # convert to floats and guard division
        with np.errstate(divide="ignore", invalid="ignore"):
            height_m = df_fe["Height"].astype(float) / 100.0
            df_fe["BMI_Calc"] = df_fe["Weight"].astype(float) / (height_m ** 2)
            # replace infinite/nans with NaN to be imputed later
            df_fe["BMI_Calc"].replace([np.inf, -np.inf], np.nan, inplace=True)

    # One-hot encode categorical values (drop_first optional)
    df_encoded = pd.get_dummies(df_fe, drop_first=True)

    # Keep separate copy of feature names (excluding target)
    if target_col not in df_encoded.columns:
        # if get_dummies changed target name or target was categorical and expanded
        # try to recover original target from original df
        if target_col in df.columns:
            # if original target was categorical, ensure it's added back
            df_encoded[target_col] = df[target_col]
        else:
            print(f"After encoding, target '{target_col}' still not present. Columns: {list(df_encoded.columns)}")
            return

    feature_cols = [c for c in df_encoded.columns if c != target_col]
    X_df = df_encoded[feature_cols].copy()
    y = df_encoded[target_col].copy()

    # -------- Preprocessing: Impute then Scale --------
    # Imputation
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X_df)
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # -------- Feature Selection: RFE with RandomForest --------
    n_features_total = X_scaled.shape[1]
    # choose 70% but at least 1 and at most n_features_total
    n_select = max(1, int(np.ceil(n_features_total * 0.7)))
    n_select = min(n_select, n_features_total)

    model = RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1)
    try:
        rfe = RFE(estimator=model, n_features_to_select=n_select, step=0.1)
        X_selected = rfe.fit_transform(X_scaled, y)
    except Exception as e:
        print(f"RFE failed: {e}. Falling back to using all features.")
        X_selected = X_scaled
        rfe = None

    # Recover selected feature names
    if rfe is not None:
        support_mask = rfe.support_
        selected_feature_names = np.array(feature_cols)[support_mask].tolist()
    else:
        selected_feature_names = feature_cols.copy()

    # Build DataFrame for selected features
    X_selected_df = pd.DataFrame(X_selected, columns=selected_feature_names)

    # -------- Feature Extraction: PCA (optional) --------
    n_components = min(10, X_selected.shape[1])
    if n_components >= 1:
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X_selected)
        pca_colnames = [f"PCA_{i+1}" for i in range(X_pca.shape[1])]
        X_pca_df = pd.DataFrame(X_pca, columns=pca_colnames)
    else:
        X_pca_df = pd.DataFrame()
        pca_colnames = []

    # -------- Final DataFrames (attach target) --------
    df_selected = pd.concat([X_selected_df.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    if not X_pca_df.empty:
        df_pca = pd.concat([X_pca_df.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    else:
        df_pca = None

    # Save outputs
    out_file_selected = os.path.join(output_dir, f"{save_name}_processed_selected.csv")
    out_file_pca = os.path.join(output_dir, f"{save_name}_processed_pca.csv")
    try:
        df_selected.to_csv(out_file_selected, index=False)
        print(f"Saved: {out_file_selected} (shape: {df_selected.shape})")
    except Exception as e:
        print(f"Failed to save selected features CSV: {e}")

    if df_pca is not None:
        try:
            df_pca.to_csv(out_file_pca, index=False)
            print(f"Saved: {out_file_pca} (shape: {df_pca.shape})")
        except Exception as e:
            print(f"Failed to save PCA CSV: {e}")
    else:
        print("PCA output was empty; no PCA file saved.")

    # Return a small report
    return {
        "input_path": file_path,
        "n_original_features": len(feature_cols),
        "n_selected_features": X_selected.shape[1],
        "pca_components": 0 if X_pca_df.empty else X_pca_df.shape[1],
        "saved_selected": out_file_selected,
        "saved_pca": out_file_pca if df_pca is not None else None
    }

# Run pipeline for all datasets
if __name__ == "__main__":
    reports = {}
    for name, file in datasets.items():
        if os.path.exists(file):
            print(f"\nProcessing '{name}' from {file} ...")
            reports[name] = preprocess_dataset(file, name)
        else:
            print(f"File not found: {file}")
    print("\nSummary reports:")
    for k, v in reports.items():
        print(k, "->", v)
