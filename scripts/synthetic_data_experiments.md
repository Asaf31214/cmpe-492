# Synthetic Data Augmentation Experiments - Pediatric PCNL Prediction

**Date:** 2026-04-30  
**Goal:** Improve failure class prediction using LLM-generated synthetic data

---

## Dataset

| Metric | Value |
|--------|-------|
| Total patients | 269 |
| Success (stone-free) | 213 (79.2%) |
| Failure (residual) | 56 (20.8%) |
| Class imbalance ratio | 3.8:1 |
| Preoperative features | 14 |

---

## Experiment 1: 50 Synthetic Failure Cases

**Setup:** Generated 50 synthetic failures with qwen3.5:4b, trained on original + synthetic

| Model | Metric | Baseline (269) | +50 Synthetic (319) | Change |
|-------|--------|----------------|---------------------|--------|
| XGBoost | Accuracy | 68.5% | 67.2% | -1.3% |
| | F1 (weighted) | 0.709 | 0.674 | -0.035 |
| | Failure Recall | 54.5% | 52.4% | -2.1% |
| SVM | Accuracy | 77.8% | 70.3% | -7.5% |
| | F1 (weighted) | 0.784 | 0.705 | -0.079 |
| | Failure Recall | 54.5% | 57.1% | +2.6% |
| Logistic | Accuracy | 59.3% | 57.8% | -1.5% |
| | F1 (weighted) | 0.632 | 0.591 | -0.041 |
| | Failure Recall | 63.6% | 57.1% | -6.5% |

**Conclusion:** No improvement with 50 samples.

---

## Experiment 2: 100 Synthetic Failure Cases

**Setup:** Generated 100 synthetic failures, trained on original + synthetic

| Model | Metric | Baseline (269) | +100 Synthetic (369) | Change |
|-------|--------|----------------|----------------------|--------|
| XGBoost | Accuracy | 68.5% | **77.0%** | **+8.5%** |
| | F1 (weighted) | 0.709 | **0.771** | **+0.062** |
| | Failure Recall | 54.5% | **74.2%** | **+19.7%** |
| SVM | Accuracy | 77.8% | **81.1%** | **+3.3%** |
| | F1 (weighted) | 0.784 | **0.812** | **+0.028** |
| | Failure Recall | 54.5% | **83.9%** | **+29.4%** |
| Logistic | Accuracy | 59.3% | 63.5% | +4.2% |
| | F1 (weighted) | 0.632 | 0.638 | +0.006 |
| | Failure Recall | 63.6% | 67.7% | +4.1% |

**Conclusion:** Significant improvement observed. **Later identified as data leakage** (synthetic samples in test set).

### Synthetic Data Characteristics (n=100)

| Feature | Synthetic Mean | Real Failure Mean |
|---------|----------------|-------------------|
| GUY SCORE | 3.28 | 2.54 |
| TOPLAM TAŞ YÜKÜ (cm²) | 4.12 | 3.67 |
| GSS 3-4 | 90% | 41% |
| Stone >3cm² | 94% | 36% |
| GSS 3-4 AND Stone >3 | 90% | 21% |

**Problem:** Synthetic failures are stereotypical (high complexity), not matching real failure diversity.

---

## Experiment 3: 200 Synthetic Failure Cases (DATA LEAKAGE DISCOVERED)

**Setup:** Generated 200 synthetic failures, trained on original + synthetic

| Model | Metric | Baseline (269) | +200 Synthetic (469) | Change |
|-------|--------|----------------|----------------------|--------|
| XGBoost | Accuracy | 68.5% | **81.9%** | **+13.4%** |
| | F1 (weighted) | 0.709 | **0.820** | **+0.111** |
| | Failure Precision | 33% | **87%** | +54% |
| | Failure Recall | 54.5% | **78.4%** | +23.9% |
| SVM | Accuracy | 77.8% | 75.5% | -2.3% |
| | F1 (weighted) | 0.784 | 0.756 | -0.028 |
| | Failure Recall | 54.5% | 74.5% | +20.0% |
| Logistic | Accuracy | 59.3% | 72.3% | +13.0% |
| | F1 (weighted) | 0.632 | 0.722 | +0.090 |
| | Failure Recall | 63.6% | 78.4% | +14.8% |

**Test set composition (FLAWED):**
- Total test: 94 samples
- Real failures in test: ~11
- **Synthetic failures in test: ~40** ← DATA LEAKAGE

**Root cause:** `train_test_split()` was applied AFTER concatenating synthetic data, so ~40 synthetic samples ended up in the test set. The model was evaluated on the same distribution it memorized during training.

---

## Experiment 4: 200 Synthetic Failure Cases (CORRECTED - No Leakage)

**Setup:** Split original data first, add synthetic ONLY to training, test on real data only

| Model | Metric | Baseline (269) | +200 Synthetic | Change |
|-------|--------|----------------|----------------|--------|
| XGBoost | Accuracy | 68.5% | **77.8%** | **+9.3%** |
| | F1 (weighted) | 0.709 | 0.758 | +0.049 |
| | Failure Precision | 33% | 43% | +10% |
| | **Failure Recall** | **54.5%** | **27.3%** | **-27.2%** ✗ |
| | Confusion Matrix | TN=31, FP=12, FN=5, TP=6 | TN=39, FP=4, FN=8, TP=3 | |
| SVM | Accuracy | 77.8% | 66.7% | -11.1% |
| | F1 (weighted) | 0.784 | 0.697 | -0.087 |
| | Failure Recall | 54.5% | 63.6% | +9.1% |
| Logistic | Accuracy | 59.3% | 59.3% | 0% |
| | F1 (weighted) | 0.632 | 0.621 | -0.011 |
| | Failure Recall | 63.6% | 27.3% | -36.3% |

**Test set (corrected):**
- Total test: 54 samples (real only)
- Failures in test: 11 (real only)

**Conclusion:** Synthetic data **does not improve** failure detection when evaluated correctly. The model becomes conservative (predicts success more often), missing actual failures.

---

## Summary of All Experiments

| Experiment | Synthetic N | Test Set | Result |
|------------|-------------|----------|--------|
| 1 | 50 | Real only | No improvement |
| 2 | 100 | **Mixed (leakage)** | False positive (+19% recall) |
| 3 | 200 | **Mixed (leakage)** | False positive (+24% recall) |
| 4 | 200 | Real only | **Worse failure recall** (-27%) |

---

## Key Findings

### 1. Data Leakage Cause
- Synthetic samples concatenated before train/test split
- ~20% of synthetic data (40 samples) leaked into test set
- Model evaluated on same distribution it memorized

### 2. LLM Synthetic Data Characteristics
- Generated failures are **stereotypical**: 90% have GSS 3-4 + large stones
- Real failures are **diverse**: only 21% have both GSS 3-4 + large stones
- LLM amplifies strongest signal, doesn't capture failure diversity

### 3. Why It Didn't Work
- 56 real failures is too few for LLM to learn meaningful patterns
- LLM interpolates existing patterns, doesn't create new medical knowledge
- Synthetic data shifts decision boundary, doesn't add generalizable signal
- With corrected evaluation, model misses more real failures

### 4. What Actually Works (from literature)
- Multi-center data collection (more real samples)
- Proper class weighting + threshold tuning
- Accept ~70% accuracy ceiling with small imbalanced data

---

## Recommendation

**Do not use LLM synthetic data augmentation for this task.** The apparent improvements were entirely due to data leakage. Correct evaluation shows:

1. No improvement in failure detection
2. Model becomes conservative, misses real failures
3. Synthetic failures don't match real failure diversity

**Alternative approaches:**
- Collect more real data (multi-center collaboration)
- Use simpler baselines with proper class weighting
- Publish as negative result (methodologically valuable)

---

## Files Generated

- `synthetic_data_gen.py` - LLM synthetic data generation script
- `train_augmented.py` - Training script with/without synthetic data
- `data/tabular/synthetic_outcome_2.csv` - 200 synthetic failure cases
- `data/tabular/llm_predictions_qwen_v3.csv` - Direct LLM prediction results
