# Ovarian Cancer Blood Methylation Classifier

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-Illumina%2027k-green.svg)](https://www.illumina.com/)
[![Dataset](https://img.shields.io/badge/GEO-GSE19711-orange.svg)](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE19711)

Blood DNA methylation classifier for ovarian cancer detection and stage prediction, built from the UK Ovarian Cancer Population Study (UKOPS). Part of a multi-disease pan-methylation diagnostic panel.

---

## Overview

| Model | Task | n | Balanced Acc | p-value | Result |
|---|---|---|---|---|---|
| **Model 1** | Control vs Pre-treatment OC | 405 | **0.753** | 0.001 | ✅ Significant |
| **Model 2** | Early (I+II) vs Late (III+IV) stage | 129 | 0.465 | 0.341 | ❌ Negative result |

**Model 2 is an important negative result** — OC stage is not detectable from whole blood methylation. This is biologically expected: stage reflects spatial tumour spread, not circulating blood composition.

---

## Dataset

**GSE19711** — UK Ovarian Cancer Population Study (UKOPS)

- **Platform:** Illumina Infinium 27k Human DNA Methylation BeadChip v1.2 (~27,578 CpGs)
- **Tissue:** Peripheral whole blood
- **Samples:** 540 total — 266 OC cases (131 pre-treatment, 135 post-treatment) + 274 controls
- **Population:** Postmenopausal women, age 49–91 years
- **Clinical data:** FIGO stage, histological subtype, grade, CA125, age at diagnosis

**Reference:** Teschendorff AE et al. *An epigenetic signature in peripheral blood predicts active ovarian cancer.* PLoS ONE 2009; and *Age-dependent DNA methylation of genes that are suppressed in stem cells is a hallmark of cancer.* Genome Research 2010.

---

## Key Design Decisions

### 1. Post-treatment exclusion
Post-treatment cases (n=135) are excluded because platinum-based chemotherapy causes widespread demethylation, creating a strong artifactual signal unrelated to cancer biology. Including them inflates performance by making cases easier to distinguish from controls. Only pre-treatment cases (n=131, blood drawn at diagnosis before any treatment) are used.

### 2. Leak-free feature selection
A critical methodological decision. Feature selection using a t-test is wrapped inside a custom `TTestSelector` transformer that fits **only on the training fold** of each cross-validation split. This prevents the held-out test sample from influencing which probes are selected — a common source of optimistic bias in omics pipelines.

```python
# Leaky approach (WRONG — features selected on all data before CV)
_, pv = ttest_ind(X[controls], X[cases])
top50 = np.argsort(pv)[:50]
X_selected = X[:, top50]
scores = cross_val_score(model, X_selected, y, cv=cv5)  # leaky!

# Leak-free approach (CORRECT — TTestSelector inside Pipeline)
pipeline = Pipeline([
    ("selector", TTestSelector(n_features=50)),  # fits on train fold only
    ("scaler",   StandardScaler()),
    ("model",    RandomForestClassifier(...))
])
scores = cross_val_score(pipeline, X, y, cv=cv5)  # leak-free
```

### 3. Outlier removal (Model 2 only)
Two samples (GSM492239, GSM492282 — stages IIa and Ic) had PC1 values >3 SD from the mean, consistent with technical hybridisation failure. Removed before stage modelling.

### 4. Covariate analysis
Age, processing batch, and bisulfite conversion efficiency were investigated as potential confounders:
- **Age:** Cases slightly older (66.4 vs 64.9 years) — minor imbalance
- **Batch:** Batches 10 and 11 contain only controls — structural confound noted
- **Conclusion:** Adding covariates as features neither improves nor degrades performance (0.751 vs 0.753), suggesting the methylation signal is robust to these confounders

---

## Results

### Model 1 — OC Detection

```
Balanced accuracy:  0.753 ± 0.051  (5-fold CV)
Permutation test:   p = 0.001  (1000 permutations)

              precision  recall  f1
Control           0.84    0.85  0.84  (n=274)
Pre-treatment OC  0.68    0.66  0.67  (n=131)

Published baseline: AUC 0.80 (Teschendorff 2010, SVM, n=491)
```

The 0.047 gap to the published result is attributable to:
- Smaller sample size (405 vs 491)
- Generic t-test feature selection vs biologically-informed polycomb gene enrichment
- Conservative 5-fold CV vs train/test split used in original paper

### Model 2 — OC Stage (Negative Result)

```
Balanced accuracy:  0.465  (LOO-CV) — below chance
Permutation test:   p = 0.341  (not significant)
```

Blood methylation does not carry stage information. This negative result is biologically meaningful and consistent with the literature — no published study has successfully predicted OC stage from blood methylation alone.

### Leakage Quantification

| Method | Balanced Acc | Inflation |
|---|---|---|
| Leak-free (nested selection) | 0.753 | — |
| Leaky (pre-selected features) | ~0.762 | +0.009 |

Leakage inflation is small for this dataset (n=405, 27k probes) but increases with smaller datasets and higher probe counts.

---

## Comparison to Literature

| Study | Method | Tissue | n | AUC/Bal Acc |
|---|---|---|---|---|
| Teschendorff 2010 | SVM, 27k array | Whole blood | 491 | AUC 0.80 |
| **This work** | RF, 27k array, leak-free | Whole blood | 405 | BA 0.753 |

---

## Clinical Context

Ovarian cancer has a 5-year survival of ~30% due to late-stage diagnosis. CA125 + transvaginal ultrasound — the current standard — has poor specificity, leading to unnecessary surgery. A blood methylation test could complement CA125:

- **CA125 alone:** Sensitivity ~70% at 95% specificity (pre-clinical UKCTOCS data)
- **Methylation alone:** Balanced accuracy 0.753 (this work)
- **Combined (projected):** De Borre et al. 2024 showed cfDNA methylation + CA125 achieves 94.4% sensitivity for high-risk cancers

CA125 values are available in this dataset — combining them with methylation is a natural next step.

---

## Repository Structure

```
Ovarian_Cancer_Data/
└── GSE19711_series_matrix.txt.gz     ← Download from GEO

preeclampsia_data/saved_models/
├── rf_ovarian_cancer.pkl             ← Model 1 trained weights
├── rf_oc_stage.pkl                   ← Model 2 trained weights
├── X_oc.npy                          ← Feature matrix (405 × 27537)
└── y_oc.npy                          ← Labels (405,)
```

---

## Requirements

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn joblib
```

| Package | Version | Purpose |
|---|---|---|
| pandas | ≥1.5 | Data manipulation |
| numpy | ≥1.23 | Numerical computation |
| scipy | ≥1.9 | t-test feature selection |
| scikit-learn | ≥1.2 | Pipeline, RF, CV, permutation test |
| matplotlib / seaborn | ≥3.6 | Visualisation |
| joblib | ≥1.2 | Model serialisation |

---

## Usage

### Download data
```
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE19711
```
Download the Series Matrix file and save to `Ovarian_Cancer_Data/`.

### Run notebook
```bash
jupyter notebook Methylomic_Ovarian_Cancer_Final.ipynb
```

### Load saved model
```python
import joblib
import numpy as np

rf_oc = joblib.load("preeclampsia_data/saved_models/rf_ovarian_cancer.pkl")

# X_new: (n_samples, 27537) — imputed beta values, all probes in order
# TTestSelector inside the pipeline handles feature selection automatically
y_pred = rf_oc.predict(X_new)        # 0 = Control, 1 = OC
y_prob = rf_oc.predict_proba(X_new)  # probability scores
```

---

## Limitations

- **Whole blood, not cfDNA** — blood cell methylation reflects immune cell composition changes due to tumour presence; cfDNA-based approaches may offer earlier detection
- **Post-treatment cases excluded** — reduces training data; future work should model post-treatment methylation separately as a treatment response predictor
- **Stage not detectable** — a fundamental biological limitation, not a modelling failure
- **No independent validation cohort** — all evaluation is cross-validation on GSE19711; prospective validation required before clinical use
- **27k array** — older platform covering only ~27,000 CpGs; EPIC array (~850,000 CpGs) may reveal additional signal

---

## Multi-Disease Panel Context

This notebook is part of a larger pan-methylation diagnostic panel:

| Model | Platform | Tissue | Balanced Acc | p-value |
|---|---|---|---|---|
| PE detection | 450k | Placenta | 0.857 | 0.001 |
| PE severity | 450k | Placenta | 0.803 | 0.001 |
| GDM detection | 450k | Placenta | 0.904 | TBD |
| **OC detection** | **27k** | **Blood** | **0.753** | **0.001** |
| OC stage | 27k | Blood | N/A | 0.341 |

---

## Citation

```
Naik, A. (2026). Ovarian Cancer Blood Methylation Classifier.
GitHub. https://github.com/aniketnaik7/Methylomic-Data-Analysis

Data: Teschendorff AE et al. An epigenetic signature in peripheral
blood predicts active ovarian cancer. PLoS ONE 2009;4:e8274.
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
