# 1. Load Packages
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# 2. Load Dataset Paths
datasets = {
    "Hepatitis": "U:/pcss23/project/data/preprocess/hepatitis_preprocessed.csv",
    "Heart Disease": "U:/pcss23/project/data/preprocess/heart_preprocessed.csv",
    "Diabetes": "U:/pcss23/project/data/preprocess/diabetes_preprocessed.csv",
    "Liver Disease": "U:/pcss23/project/data/preprocess/liver_preprocessed.csv",
    "Lung Cancer": "U:/pcss23/project/data/preprocess/lung_cancer_preprocessed.csv"
}

# 3. Define Hyperparameter Grids
param_grids = {
    "rf": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    },
    "svm": {
        "C": [0.1, 1, 10],
        "gamma": ["scale", "auto"],
        "kernel": ["rbf"]
    },
    "knn": {
        "n_neighbors": [3, 5, 7],
        "weights": ["uniform", "distance"]
    },
    "xgb": {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2]
    }
}

# 4. Function: Tune Models
def tune_model(estimator, param_grid, X_train, y_train):
    """
    Runs GridSearchCV on given estimator and returns best_estimator_.
    """
    grid = GridSearchCV(estimator, param_grid, cv=3, scoring="accuracy", n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    return grid.best_estimator_

# 5. Function: Run Stacking with Tuning (with Confusion Matrix)
def run_stacking_with_tuning(dataset_path, target_column="disease_label"):
    print(f"\nLoading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)

    # Basic safety check
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in {dataset_path} columns: {df.columns.tolist()}")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=30, stratify=y
    )

    # Tune each base learner
    tuned_rf = tune_model(RandomForestClassifier(random_state=42), param_grids["rf"], X_train, y_train)
    tuned_svm = tune_model(SVC(probability=True, random_state=42), param_grids["svm"], X_train, y_train)
    tuned_knn = tune_model(KNeighborsClassifier(), param_grids["knn"], X_train, y_train)
    tuned_xgb = tune_model(
        XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
        param_grids["xgb"], X_train, y_train
    )

    # Base learners
    base_learners = [
        ("rf", tuned_rf),
        ("svm", tuned_svm),
        ("knn", tuned_knn),
        ("xgb", tuned_xgb)
    ]

    # Meta learner
    meta_learner = LogisticRegression(max_iter=1000, random_state=42)

    # Stacking model
    stacking_model = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1
    )

    # Train and evaluate
    stacking_model.fit(X_train, y_train)
    y_pred = stacking_model.predict(X_test)

    # Determine if problem is binary or multiclass
    unique_labels = pd.Series(y_test).nunique()
    if unique_labels <= 2:
        average_mode = None  # default (binary) - metrics will work without average
    else:
        average_mode = "weighted"  # better for class imbalance in multiclass

    # Compute metrics (handle binary vs multiclass)
    results = {
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, average=average_mode) if average_mode else precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred, average=average_mode) if average_mode else recall_score(y_test, y_pred), 4),
        "F1-Score": round(f1_score(y_test, y_pred, average=average_mode) if average_mode else f1_score(y_test, y_pred), 4),
        "Confusion Matrix": confusion_matrix(y_test, y_pred)
    }

    print("\nConfusion Matrix:\n", results["Confusion Matrix"])
    return results

# 6. Run on All Datasets
final_results = {}
for name, path in datasets.items():
    print(f"\n=== Running Tuned Stacking on {name} Dataset ===")
    try:
        final_results[name] = run_stacking_with_tuning(path, target_column="disease_label")
    except Exception as e:
        print(f"Error processing {name}: {e}")
        final_results[name] = {"error": str(e)}

# 7. Display Final Results
results_df = pd.DataFrame(final_results).T
print("\nStacking with Hyperparameter Tuning Results Across Datasets ===\n")
print(results_df.to_string())
