import polars as pl
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

DATA_DIR = Path(__file__).parent / "data" / "tabular"

PREOP_COLUMNS = [
    "YAŞ",
    "CİNSİYET",
    "TARAF",
    "ASA SKORU",
    "BT",
    "LOKALİZASYON",
    "ÖZGEÇMİŞ",
    "GEÇİRİLMİŞ CERRAHİ",
    "TAS ANAMNEZI",
    "İKAB",
    "KREATİNİN",
    "SOLİTER BB",
    "GUY SCORE",
    "TOPLAM TAŞ YÜKÜ (CM2)",
]


def load_data(synthetic_files: list[str] | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load original and optionally synthetic data."""
    original = pl.read_csv(DATA_DIR / "processed.csv")

    if synthetic_files:
        synthetic_dfs = []
        for f in synthetic_files:
            path = DATA_DIR / f
            if path.exists():
                df = pl.read_csv(path)
                synthetic_dfs.append(df)
                print(f"Loaded {len(df)} synthetic samples from {f}")

        if synthetic_dfs:
            synthetic = pl.concat(synthetic_dfs)
            print(f"Total synthetic samples: {len(synthetic)}")

            # Select only prediction columns from original to match synthetic
            cols_needed = PREOP_COLUMNS + ["SONUÇ-2"]
            original = original.select(cols_needed)

            # Drop 'synthetic' marker column if present
            if "synthetic" in synthetic.columns:
                synthetic = synthetic.drop("synthetic")

            # Cast both to common schema
            float_cols = ["KREATİNİN", "TOPLAM TAŞ YÜKÜ (CM2)"]
            for col in PREOP_COLUMNS:
                if col in float_cols:
                    original = original.with_columns(pl.col(col).cast(pl.Float64, strict=False))
                    if col in synthetic.columns:
                        synthetic = synthetic.with_columns(pl.col(col).cast(pl.Float64, strict=False))
                elif col in original.columns:
                    original = original.with_columns(pl.col(col).cast(pl.Int64, strict=False))
                    if col in synthetic.columns:
                        synthetic = synthetic.with_columns(pl.col(col).cast(pl.Int64, strict=False))

            return original, synthetic

    return original, pl.DataFrame()


def prepare_features(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Prepare feature matrix and target vector."""
    X = df.select(PREOP_COLUMNS).to_numpy()
    y = df["SONUÇ-2"].cast(pl.Int64, strict=False).to_numpy() - 1
    return X, y


def evaluate_model(name: str, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None):
    """Print evaluation metrics."""
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    print(f"Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={tn:3d}  FP={fp:3d}")
    print(f"  FN={fn:3d}  TP={tp:3d}")

    print(f"\nClass-wise metrics:")
    print(f"  Success (class 0): precision={precision_score(y_true, y_pred, labels=[0], zero_division=0):.4f}, recall={recall_score(y_true, y_pred, labels=[0], zero_division=0):.4f}")
    print(f"  Failure (class 1): precision={precision_score(y_true, y_pred, labels=[1], zero_division=0):.4f}, recall={recall_score(y_true, y_pred, labels=[1], zero_division=0):.4f}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
    """Train XGBoost with class imbalance handling."""
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    scale_pos_weight = float(n_neg) / float(n_pos) if n_pos > 0 else 1.0

    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.9,
        min_child_weight=3,
        scale_pos_weight=scale_pos_weight,
        random_state=492,
        eval_metric="logloss",
        tree_method="hist",
    )

    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    y_proba = xgb_model.predict_proba(X_test)

    return evaluate_model("XGBoost", y_test, y_pred, y_proba)


def train_svm(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
    """Train SVM with RBF kernel."""
    svm_model = SVC(
        C=100,
        gamma=0.1,
        kernel="rbf",
        class_weight="balanced",
        random_state=492,
        probability=True,
    )

    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    y_proba = svm_model.predict_proba(X_test)

    return evaluate_model("SVM", y_test, y_pred, y_proba)


def train_logistic(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
    """Train Logistic Regression."""
    lr_model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=492,
    )

    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    y_proba = lr_model.predict_proba(X_test)

    return evaluate_model("Logistic Regression", y_test, y_pred, y_proba)


def cross_validate(X: np.ndarray, y: np.ndarray, model_name: str = "XGBoost"):
    """Perform stratified cross-validation."""
    n_neg = np.sum(y == 0)
    n_pos = np.sum(y == 1)
    scale_pos_weight = float(n_neg) / float(n_pos) if n_pos > 0 else 1.0

    if model_name == "XGBoost":
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.9,
            min_child_weight=3,
            scale_pos_weight=scale_pos_weight,
            random_state=492,
            eval_metric="logloss",
            tree_method="hist",
        )
    elif model_name == "SVM":
        model = SVC(C=100, gamma=0.1, kernel="rbf", class_weight="balanced", random_state=492)
    else:
        model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=492)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=492)
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1_weighted")

    print(f"\n5-Fold CV F1 (weighted): {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    return scores.mean(), scores.std()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train classifiers on original + synthetic data")
    parser.add_argument("--synthetic", type=str, nargs="*", default=None, help="Synthetic data files to include")
    parser.add_argument("--no-synthetic", action="store_true", help="Train on original data only")
    parser.add_argument("--cv", action="store_true", help="Run cross-validation")
    parser.add_argument("--model", type=str, choices=["xgb", "svm", "lr", "all"], default="all", help="Model to train")

    args = parser.parse_args()

    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    original, synthetic = load_data(args.synthetic if not args.no_synthetic else None)

    print(f"\nOriginal data: {len(original)} samples")
    print(f"  Success: {len(original.filter(pl.col('SONUÇ-2') == 1))}")
    print(f"  Failure: {len(original.filter(pl.col('SONUÇ-2') == 2))}")

    # Split ORIGINAL data first (test set must be real only)
    X_orig, y_orig = prepare_features(original)
    np.random.seed(492)
    X_train_orig, X_test, y_train_orig, y_test = train_test_split(
        X_orig, y_orig, test_size=0.2, random_state=492, stratify=y_orig
    )

    if len(synthetic) > 0:
        print(f"Synthetic data: {len(synthetic)} samples")
        # Add synthetic only to training set
        X_synth, y_synth = prepare_features(synthetic)
        X_train = np.vstack([X_train_orig, X_synth])
        y_train = np.concatenate([y_train_orig, y_synth])
        print(f"Combined training: {len(X_train)} samples")
    else:
        X_train, y_train = X_train_orig, y_train_orig

    print(f"Test set (real only): {len(X_test)} samples")
    print(f"  Test failures: {np.sum(y_test == 1)}")

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    if args.cv:
        print(f"\n{'='*60}")
        print("CROSS-VALIDATION")
        print(f"{'='*60}")
        cross_validate(X_train_scaled, y_train, "XGBoost")
        cross_validate(X_train_scaled, y_train, "SVM")
        cross_validate(X_train_scaled, y_train, "LR")

    print(f"\n{'='*60}")
    print("TRAINING MODELS")
    print(f"{'='*60}")

    models = {
        "xgb": train_xgboost,
        "svm": train_svm,
        "lr": train_logistic,
    }

    if args.model == "all":
        for name, trainer in models.items():
            trainer(X_train_scaled, y_train, X_test_scaled, y_test)
    elif args.model in models:
        models[args.model](X_train_scaled, y_train, X_test_scaled, y_test)


if __name__ == "__main__":
    main()
