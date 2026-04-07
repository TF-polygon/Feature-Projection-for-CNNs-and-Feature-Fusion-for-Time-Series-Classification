# Feature Projection for CNNs and Feature Fusion for Time Series Classification

> **Research Status: Incomplete / Work in Progress**
> This repository contains research code for an unfinished study on unsupervised pattern discovery and image-based classification of financial time series data.

---

## Overview

Financial time series data — such as Forex price sequences — does not come with ground-truth labels. This research proposes a full pipeline to address this challenge through two major stages:

1. **Pseudo-labeling via unsupervised clustering** — Latent patterns embedded in unlabeled time series are captured using DTW-based K-Means clustering, and each window is assigned a pseudo-label (cluster ID) to serve as a classification target.
2. **Multi-feature image fusion CNN** — Each labeled time series window is converted into three distinct 2D image representations (GASF, GADF, RP), and a CNN model (`muffin`) is trained by fusing all three feature projections as multi-channel inputs.

### Target Data

| Symbol | Timeframe |
|--------|-----------|
| EUR/USD | 1H |
| GBP/USD | 1H |
| USD/CAD | 1H |
| USD/JPY | 1H |

---

## Research Pipeline
<p align="center">
  <img src="https://github.com/user-attachments/assets/2be1e245-e809-46b6-8ce3-0cabfeaed3e9"><br>
  <b>Figure 1.</b> Overview of the proposed framework.
</p>

---

## Repository Structure

```
.
├── data/                         # Raw & processed CSV data
│   └── processed/                # Pseudo-labeled datasets
├── datasets/
│   └── Sample/                   # Sample image datasets
├── images/                       # Generated image outputs
├── src/
│   └── muffin/                   # CNN model package (Multi-feature Fusion)
├── elbow_method.py               # Step 1: Optimal k & window search
├── clustering.py                 # Step 2: DTW K-Means clustering
├── mapping.py                    # Step 3: Pseudo-label mapping (single)
├── auto_mapping.py               # Step 3: Pseudo-label mapping (batch)
├── image_converter.py            # Step 4: Time series → image (single)
├── auto_image_converter.py       # Step 4: Time series → image (batch)
├── predict_clusters.py           # Cluster prediction utility
├── main.py                       # Step 5: CNN training entry point
├── pyproject.toml                # Package config for muffin
└── README.md
```

---

## Step-by-Step Usage

### Step 1 — Find Optimal Clustering Conditions

Evaluate WCSS (inertia), Silhouette Coefficient, Calinski-Harabasz Index, and Davies-Bouldin Index across `k = 2..15` for a given window shape.

```bash
python elbow_method.py \
  --path data/EURUSD_1H.csv \
  --window_shape 48 \
  --stride 12
```

Results are saved to `experimental_data/<window_shape>_elbow_method_results_<timestamp>.csv`.

Repeat for all window shapes `{24, 48, 72, 96}` and compare metrics across currency pairs to determine the optimal `(window_shape, k)` combination.

---

### Step 2 — Run DTW K-Means Clustering (Training Data)

Once the optimal conditions are identified, run clustering to generate pseudo-labeled training data.

```bash
python clustering.py \
  --path data/EURUSD_1H_train.csv \
  --n_clusters 5 \
  --window_shape 48 \
  --symbol EURUSD \
  --stride 12
```

Outputs:
- `data/processed/clustered_EURUSD_5cls.csv`
- `clustering_data/EURUSD_5k.npz`
- `clustering_data/model/EURUSD_5k.joblib`

---

### Step 3 — Map Pseudo-Labels to Validation / Test Data

Apply the trained clustering model to validation and test splits.

```bash
python mapping.py \
  --model_path clustering_data/model/EURUSD_5k.joblib \
  --data_path data/EURUSD_1H_valid.csv \
  --symbol EURUSD \
  --n_clusters 5 \
  --window_size 48 \
  --data_type valid
```

---

### Step 4 — Convert Time Series Windows to Images

Convert each pseudo-labeled window into GASF, GADF, and RP images.

```bash
# GASF
python image_converter.py \
  --path data/processed/clustered_EURUSD_5cls_train.csv \
  --image_type GASF \
  --data_type train \
  --num_classes 5 \
  --symbol EURUSD

# GADF
python image_converter.py \
  --path data/processed/clustered_EURUSD_5cls_train.csv \
  --image_type GADF \
  --data_type train \
  --num_classes 5 \
  --symbol EURUSD

# RP
python image_converter.py \
  --path data/processed/clustered_EURUSD_5cls_train.csv \
  --image_type RP \
  --data_type train \
  --num_classes 5 \
  --symbol EURUSD
```

Images are saved under:
```
datasets/P-FXImageSet/<k>k/<data_type>/<symbol>/<image_type>/<class_id>/
```

---

### Step 5 — Train the Multi-Feature Fusion CNN

```bash
python main.py \
  --dataset datasets/P-FXImageSet/5k \
  --input_size 48 \
  --num_features 3 \
  --epochs 50 \
  --batch_size 32 \
  --num_classes 5
```

The `muffin` package (located in `src/muffin/`) implements the multi-branch CNN that accepts GASF, GADF, and RP images simultaneously, fuses their feature maps, and outputs class probabilities.

---

## Methodology

### Pseudo-Labeling via DTW K-Means

Since Forex time series have no ground-truth pattern labels, this study uses **unsupervised clustering** to discover recurring market regimes. Before clustering, each window is smoothed using **Lowess smoothing** (via `tsmoothie`) and normalized to `[0, 1]` via Min-Max scaling to make patterns scale-invariant.

Clustering is performed with `TimeSeriesKMeans` from `tslearn` using **Dynamic Time Warping (DTW)** as the distance metric, which is robust to temporal shifts and distortions in financial patterns.

The optimal `(window_shape, k)` pair is selected by jointly minimizing **DBI** and maximizing the **Silhouette Coefficient**, cross-referenced with the WCSS elbow point.

| Metric | Optimum |
|--------|---------|
| WCSS (Inertia) | Elbow point |
| Silhouette Coefficient | Higher is better |
| Davies-Bouldin Index | Lower is better |
| Calinski-Harabasz Index | Higher is better |

### Image Encoding

Each 1D time series window of length `w` is encoded into a `w × w` 2D image using three different transformations from `pyts`:

| Method | Description |
|--------|-------------|
| **GASF** | Gramian Angular Summation Field — encodes temporal correlations via angular cosine sum |
| **GADF** | Gramian Angular Difference Field — captures directional temporal differences |
| **RP** | Recurrence Plot — visualizes the recurrence of states in phase space |

<p align="center">
  <img width="1647" height="1347" alt="image conversion" src="https://github.com/user-attachments/assets/e6ba404d-cc67-4528-bdad-38a2bea8eaa2"><br>
  <b>Figure 2.</b> Visualization of the three image conversion scheme used. Each row demonstrates how temporal features are mapped into a standardized 2D space (normalized between -1.0 and 1.0).
</p>

These three representations capture complementary aspects of the same time series segment, motivating their fusion.

### Multi-Feature Fusion CNN (`muffin`)

The `muffin` model (Multi-Feature Fusion) is a multi-branch convolutional architecture where each branch independently processes one image type (GASF, GADF, RP). Branch feature maps are fused (concatenated) before the classification head.

---

## Dependencies

```bash
pip install tslearn tsmoothie pyts scikit-learn pandas numpy matplotlib rich tqdm joblib
```

To install the `muffin` package locally:

```bash
pip install -e .
```

---

## Clustering Hyperparameter Search Space

| Parameter | Values |
|-----------|--------|
| `window_shape` | 24, 48, 72, 96 |
| `n_clusters` (k) | 2 – 15 |
| `stride` | 12 |
| `smooth_fraction` (Lowess) | 0.6 |
| `n_init` | 3 |
| `max_iter` | 20 |
| `random_state` | 123 |

---

## Notes

- This research was not completed. The CNN training code (`src/muffin/`) exists but experimental results and final model evaluations are absent.
- The pseudo-labeling approach is fully unsupervised — no market direction labels (buy/sell/hold) are used. Cluster IDs reflect discovered structural patterns in price movement, not financial outcomes.
- All image datasets are derived solely from the **Close price** column of the input OHLCV CSV files.
