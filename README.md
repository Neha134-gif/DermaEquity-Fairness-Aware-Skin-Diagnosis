# DermaEquity: Empirical Baseline Audit on Skin-Tone Diagnostic Disparities

An independent empirical investigation evaluating diagnostic performance disparities and class-imbalance artifacts in standard deep learning models across Fitzpatrick skin tones using the Fitzpatrick17k benchmark dataset.

---

## 📌 Project Overview
Computer-aided dermatology models often struggle to generalize across diverse demographic cohorts. Before introducing fairness interventions, this repository establishes an **experimental baseline audit** using an off-the-shelf convolutional architecture (MobileNetV3-Large) trained without fairness constraints to uncover:
1. The real-world diagnostic performance gap between Light Skin (Fitzpatrick I–III) and Dark/Indian Skin (Fitzpatrick IV–VI).
2. The **Accuracy Paradox** caused by extreme class imbalance in dermatological benchmarks.

---

## 📁 Repository Structure

```text
├── model_py.ipynb                 # Complete end-to-end Google Colab experimental notebook
├── existing_model_bias_proof.png   # Side-by-side Confusion Matrix comparison plot
└── README.md                      # Project documentation & empirical findings
```

- **`model_py.ipynb`**: Contains complete executable code for cloning Fitzpatrick17k metadata, multithreaded sample image ingestion, dataset stratification, PyTorch MobileNetV3 transfer learning, and skin-tone stratified test evaluation.
- **`existing_model_bias_proof.png`**: High-resolution dual confusion matrix comparing predictions on Light Skin (Blue) vs. Dark Skin (Red).

---

## 🔬 Dataset & Experimental Setup

- **Benchmark Dataset:** [Fitzpatrick17k](https://github.com/mattgroh/fitzpatrick17k) (16,577 annotated clinical images).
- **Tested Image Subset:** 228 verified clinical photographs downloaded via multithreaded requests.
- **Stratified Demographic Cohorts:**
  - **Light Skin Cohort:** Fitzpatrick Skin Types I, II, and III (69.1% of global dataset).
  - **Dark / Indian Skin Cohort:** Fitzpatrick Skin Types IV, V, and VI (30.9% of global dataset).
- **Classification Task:** 3 Broad Clinical Partitions (`non-neoplastic`, `benign`, `malignant`).
- **Data Partitioning:** 75% Training (171 images) | 25% Testing (57 images).
- **Backbone Architecture:** Pre-trained `MobileNetV3-Large` (ImageNet initial weights, 5.4M parameters).
- **Optimization Parameters:** Standard Cross-Entropy Loss, Adam Optimizer ($\text{lr} = 3 \times 10^{-4}$), Batch Size = 16, 5 Epochs executed on Nvidia Tesla T4 GPU.

---

## 📊 Key Findings & The "Accuracy Paradox"

Evaluation across the 57 unseen test images yielded the following empirical metrics:

| Demographic Cohort | Test Samples Evaluated | Observed Test Accuracy | Key Failure Mode Observed |
| :--- | :---: | :---: | :--- |
| **Light Skin (Types I–III)** | 17 images | **64.71%** (11 / 17) | Balanced error distribution across clinical classes. |
| **Dark / Indian Skin (Types IV–VI)** | 40 images | **87.50%** (35 / 40) | Severe **Class Imbalance Artifact** (Majority class overfitting). |

### 🔍 In-Depth Limitation Analysis:
While nominal accuracy on Dark Skin appeared higher (87.50%), deep inspection of the **Confusion Matrix** revealed a critical diagnostic failure:
1. **Majority Class Overfitting:** Out of 40 dark skin test images, 37 samples belonged strictly to the `non-neoplastic` (inflammatory) category.
2. **Failure on Critical Conditions:** The standard baseline model learned to predict `non-neoplastic` for almost all dark skin inputs. It correctly matched 34 inflammatory cases purely by frequency, but completely failed to detect `benign` and `malignant` lesions.
3. **Conclusion:** Standard accuracy is a deceptive metric in biased medical datasets. Off-the-shelf baseline models fail to learn robust pathological features on darker skin tones without explicit fairness-aware re-weighting or color constancy normalization.

---

## 🛠️ How to Replicate

1. Open `model_py.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Enable GPU acceleration: `Runtime` > `Change runtime type` > `T4 GPU`.
3. Run all cells sequentially. The notebook will automatically:
   - Clone Fitzpatrick17k metadata.
   - Fetch sample clinical images.
   - Train MobileNetV3 for 5 epochs.
   - Generate and display the dual confusion matrix audit.

---

## 💡 Next Steps for DermaEquity
Based on these baseline findings, the next phase focuses on:
- [ ] Implementing **Shades of Gray color constancy** to normalize illumination variations.
- [ ] Applying **Group-Weighted Focal Loss** to counteract majority class bias on dark skin.
- [ ] Conducting a **Grad-CAM++ explainability audit** to measure gradient leakage on background melanin.
