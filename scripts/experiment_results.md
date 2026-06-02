# Synthetic Data Experiments - Results Tables

## Dataset Overview

| Metric | Value |
|--------|-------|
| Total patients | 269 |
| Success | 213 (79.2%) |
| Failure | 56 (20.8%) |
| Features | 14 |

---

## Experiment 1: 50 Synthetic Samples (DATA LEAKAGE)

| Model | Metric | Baseline | +50 Synthetic |
|-------|--------|----------|---------------|
| XGBoost | Accuracy | 68.5% | 67.2% |
| | F1 | 0.709 | 0.674 |
| | Failure Recall | 54.5% | 52.4% |
| SVM | Accuracy | 77.8% | 70.3% |
| | F1 | 0.784 | 0.705 |
| | Failure Recall | 54.5% | 57.1% |
| Logistic | Accuracy | 59.3% | 57.8% |
| | F1 | 0.632 | 0.591 |
| | Failure Recall | 63.6% | 57.1% |

**Test set:** 64 samples (included ~10 synthetic)

---

## Experiment 2: 100 Synthetic Samples (DATA LEAKAGE)

| Model | Metric | Baseline | +100 Synthetic |
|-------|--------|----------|----------------|
| XGBoost | Accuracy | 68.5% | 77.0% |
| | F1 | 0.709 | 0.771 |
| | Failure Recall | 54.5% | 74.2% |
| SVM | Accuracy | 77.8% | 81.1% |
| | F1 | 0.784 | 0.812 |
| | Failure Recall | 54.5% | 83.9% |
| Logistic | Accuracy | 59.3% | 63.5% |
| | F1 | 0.632 | 0.638 |
| | Failure Recall | 63.6% | 67.7% |

**Test set:** 64 samples (included ~20 synthetic)

---

## Experiment 3: 200 Synthetic Samples (DATA LEAKAGE)

| Model | Metric | Baseline | +200 Synthetic |
|-------|--------|----------|----------------|
| XGBoost | Accuracy | 68.5% | 81.9% |
| | F1 | 0.709 | 0.820 |
| | Failure Precision | 33% | 87% |
| | Failure Recall | 54.5% | 78.4% |
| | Confusion Matrix | TN=31, FP=12, FN=5, TP=6 | TN=37, FP=6, FN=11, TP=40 |
| SVM | Accuracy | 77.8% | 75.5% |
| | F1 | 0.784 | 0.756 |
| | Failure Recall | 54.5% | 74.5% |
| Logistic | Accuracy | 59.3% | 72.3% |
| | F1 | 0.632 | 0.722 |
| | Failure Recall | 63.6% | 78.4% |

**Test set:** 94 samples (included ~40 synthetic)

---

## Experiment 4: 200 Synthetic Samples (CORRECTED - No Leakage)

| Model | Metric | Baseline | +200 Synthetic |
|-------|--------|----------|----------------|
| XGBoost | Accuracy | 68.5% | 77.8% |
| | F1 | 0.709 | 0.758 |
| | Failure Precision | 33% | 43% |
| | Failure Recall | 54.5% | 27.3% |
| | Confusion Matrix | TN=31, FP=12, FN=5, TP=6 | TN=39, FP=4, FN=8, TP=3 |
| SVM | Accuracy | 77.8% | 66.7% |
| | F1 | 0.784 | 0.697 |
| | Failure Recall | 54.5% | 63.6% |
| Logistic | Accuracy | 59.3% | 59.3% |
| | F1 | 0.632 | 0.621 |
| | Failure Recall | 63.6% | 27.3% |

**Test set:** 54 samples (real only)

---

## Synthetic Data Statistics

| Feature | Real Failures (n=56) | Synthetic (n=200) |
|---------|---------------------|-------------------|
| GUY SCORE mean | 2.54 | 3.35 |
| TOPLAM TAŞ YÜKÜ mean (cm²) | 3.67 | 4.07 |
| GSS 3-4 | 41% | 90% |
| Stone >3cm² | 36% | 85% |
