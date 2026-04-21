import os
import json
import warnings
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


warnings.filterwarnings("ignore")


# =========================
# CONFIG
# =========================
DATA_PATH = "credit_risk_dataset.csv"
TARGET_COLUMN = "loan_status"
MODEL_DIR = "artifacts"
RANDOM_STATE = 42
TEST_SIZE = 0.25

# You can change this later based on recall/precision tradeoff
CUSTOM_THRESHOLD = 0.35


# =========================
# DATA LOADING
# =========================
def load_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset not found at: {file_path}\n"
            f"Please place 'credit_risk_dataset.csv' in the same folder as this script."
        )

    df = pd.read_csv(file_path)
    return df


def basic_data_report(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("DATASET OVERVIEW")
    print("=" * 80)
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nData Types:")
    print(df.dtypes)

    print("\nTarget Distribution:")
    print(df[TARGET_COLUMN].value_counts(dropna=False))
    print("\nTarget Ratio:")
    print(df[TARGET_COLUMN].value_counts(normalize=True, dropna=False))


# =========================
# FEATURE / TARGET SPLIT
# =========================
def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return X, y


# =========================
# PREPROCESSOR
# =========================
def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, list, list]:
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )

    return preprocessor, numerical_cols, categorical_cols


# =========================
# MODEL BUILDERS
# =========================
def build_models(preprocessor: ColumnTransformer) -> Dict[str, Pipeline]:
    models = {
        "KNN": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", KNeighborsClassifier()),
            ]
        ),
        "LogisticRegression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "SVC": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", SVC(probability=True, class_weight="balanced")),
            ]
        ),
        "DecisionTree": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
        "RandomForest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )),
            ]
        ),
    }

    return models


# =========================
# EVALUATION
# =========================
def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
    model_name: str = "Model",
) -> Dict[str, float]:
    y_pred_default = model.predict(X_test)

    if hasattr(model.named_steps["model"], "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        raise ValueError(f"{model_name} does not support predict_proba().")

    y_pred_custom = (y_prob >= threshold).astype(int)

    results = {
        "model": model_name,
        "accuracy_default": accuracy_score(y_test, y_pred_default),
        "precision_default": precision_score(y_test, y_pred_default, zero_division=0),
        "recall_default": recall_score(y_test, y_pred_default, zero_division=0),
        "f1_default": f1_score(y_test, y_pred_default, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "accuracy_custom_threshold": accuracy_score(y_test, y_pred_custom),
        "precision_custom_threshold": precision_score(y_test, y_pred_custom, zero_division=0),
        "recall_custom_threshold": recall_score(y_test, y_pred_custom, zero_division=0),
        "f1_custom_threshold": f1_score(y_test, y_pred_custom, zero_division=0),
    }

    print("\n" + "=" * 80)
    print(f"{model_name} | DEFAULT THRESHOLD (0.5)")
    print("=" * 80)
    print(classification_report(y_test, y_pred_default, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_default))

    print("\n" + "=" * 80)
    print(f"{model_name} | CUSTOM THRESHOLD ({threshold})")
    print("=" * 80)
    print(classification_report(y_test, y_pred_custom, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_custom))

    print(f"\nROC-AUC: {results['roc_auc']:.4f}")

    return results


# =========================
# MODEL COMPARISON
# =========================
def compare_baseline_models(
    models: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    comparison_rows = []

    print("\n" + "=" * 80)
    print("TRAINING BASELINE MODELS")
    print("=" * 80)

    for name, pipeline in models.items():
        print(f"\nTraining {name}...")
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        row = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0),
            "ROC_AUC": roc_auc_score(y_test, y_prob),
        }
        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows).sort_values(by="ROC_AUC", ascending=False)
    print("\nBaseline Model Comparison:")
    print(comparison_df)

    return comparison_df


# =========================
# RANDOM FOREST TUNING
# =========================
def tune_random_forest(preprocessor: ColumnTransformer) -> GridSearchCV:
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]
    )

    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [5, 10, None],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
    }

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=3,
        n_jobs=-1,
        verbose=1,
    )

    return grid


# =========================
# FEATURE IMPORTANCE
# =========================
def get_feature_names_from_preprocessor(
    fitted_preprocessor: ColumnTransformer,
    numerical_cols: list,
    categorical_cols: list,
) -> list:
    cat_encoder = fitted_preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()
    feature_names = numerical_cols + cat_feature_names
    return feature_names


def save_feature_importance_plot(
    best_model: Pipeline,
    numerical_cols: list,
    categorical_cols: list,
    output_path: str,
    top_n: int = 15,
) -> None:
    preprocessor = best_model.named_steps["preprocessor"]
    rf_model = best_model.named_steps["model"]

    feature_names = get_feature_names_from_preprocessor(preprocessor, numerical_cols, categorical_cols)
    importances = rf_model.feature_importances_

    feature_importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values(by="importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df["feature"][::-1], feature_importance_df["importance"][::-1])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top Feature Importances - Random Forest")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"\nFeature importance plot saved to: {output_path}")
    print("\nTop Features:")
    print(feature_importance_df)


# =========================
# ROC CURVE
# =========================
def save_roc_curve_plot(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: str,
) -> None:
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"ROC curve saved to: {output_path}")


# =========================
# SAVE ARTIFACTS
# =========================
def save_artifacts(
    best_model: Pipeline,
    comparison_df: pd.DataFrame,
    metrics_dict: Dict[str, float],
    model_dir: str,
) -> None:
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "credit_risk_model.joblib")
    comparison_path = os.path.join(model_dir, "model_comparison.csv")
    metrics_path = os.path.join(model_dir, "evaluation_metrics.json")

    joblib.dump(best_model, model_path)
    comparison_df.to_csv(comparison_path, index=False)

    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)

    print("\nSaved artifacts:")
    print(f"- Model: {model_path}")
    print(f"- Comparison CSV: {comparison_path}")
    print(f"- Metrics JSON: {metrics_path}")


# =========================
# MAIN
# =========================
def main() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Load data
    df = load_data(DATA_PATH)
    basic_data_report(df)

    # 2. Split X and y
    X, y = split_features_target(df)

    # 3. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print("\nTrain shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    # 4. Build preprocessor
    preprocessor, numerical_cols, categorical_cols = build_preprocessor(X)

    # 5. Baseline models
    models = build_models(preprocessor)
    comparison_df = compare_baseline_models(models, X_train, y_train, X_test, y_test)

    # 6. Tune Random Forest
    print("\n" + "=" * 80)
    print("TUNING RANDOM FOREST")
    print("=" * 80)

    rf_grid = tune_random_forest(preprocessor)
    rf_grid.fit(X_train, y_train)

    best_model = rf_grid.best_estimator_

    print("\nBest Parameters:")
    print(rf_grid.best_params_)

    # 7. Evaluate best model
    metrics_dict = evaluate_model(
        model=best_model,
        X_test=X_test,
        y_test=y_test,
        threshold=CUSTOM_THRESHOLD,
        model_name="Tuned RandomForest",
    )

    # 8. Save plots
    save_feature_importance_plot(
        best_model=best_model,
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        output_path=os.path.join(MODEL_DIR, "feature_importance.png"),
        top_n=15,
    )

    save_roc_curve_plot(
        model=best_model,
        X_test=X_test,
        y_test=y_test,
        output_path=os.path.join(MODEL_DIR, "roc_curve.png"),
    )

    # 9. Save model + outputs
    save_artifacts(
        best_model=best_model,
        comparison_df=comparison_df,
        metrics_dict=metrics_dict,
        model_dir=MODEL_DIR,
    )

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print("Project upgraded successfully.")
    print("Next step: deploy the saved joblib model using FastAPI or Streamlit.")


if __name__ == "__main__":
    main()