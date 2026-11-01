# DermaEquity: Working Prototype & Empirical Feasibility Benchmark

An empirical prototype and bias mitigation benchmark evaluating diagnostic disparities, class-imbalance artifacts, and visual explainability faithfulness across Fitzpatrick skin tones using the Fitzpatrick17k benchmark dataset.

---

## 📌 Prototype Scope & Objective
Before launching extensive full-scale model training across thousands of medical images, this repository serves as an **end-to-end working prototype (Proof-of-Concept)**. 

The primary objective of this pilot implementation is to empirically verify:
1. **Pipeline Feasibility:** Whether standard off-the-shelf convolutional architectures (MobileNetV3) exhibit measurable demographic disparities and class-imbalance artifacts on darker skin tones when trained with standard cross-entropy loss.
2. **Methodological Validity:** Whether our proposed interventions—specifically **physics-grounded color constancy (Shades of Gray)** and **class-balanced re-weighting**—can effectively mitigate this behavior without computational bloat.
3. **Visual Faithfulness (XAI):** Whether Grad-CAM++ saliency heatmaps on darker skin tones can be redirected away from confounding background melanin and locked onto true pathological lesion borders.

---

## 📁 Repository Structure

```text
├── model_py.ipynb                     # Complete end-to-end Google Colab experimental prototype notebook
├── existing_model_bias_proof.png       # Baseline limitation audit: Light vs. Dark skin confusion matrices
├── dermaequity_before_after_proof.png  # Comparative proof: Baseline flaw vs. DermaEquity balanced detection
├── gradcam_explainability_proof.png    # 3-Panel visual proof: Original image vs. Baseline leakage vs. DermaEquity localized focus
└── README.md                          # Prototype documentation, empirical findings & roadmap
```

---

## 🔬 Dataset & Experimental Prototype Setup

- **Benchmark Dataset:** [Fitzpatrick17k](https://github.com/mattgroh/fitzpatrick17k) (16,577 annotated clinical images).
- **Curated Sample Cohort:** 228 verified clinical photographs downloaded via multithreaded requests to test prototype logic.
- **Stratified Demographic Cohorts:**
  - **Light Skin Cohort:** Fitzpatrick Skin Types I, II, and III (69.1% of global dataset).
  - **Dark / Indian Skin Cohort:** Fitzpatrick Skin Types IV, V, and VI (30.9% of global dataset).
- **Classification Task:** 3 Broad Clinical Partitions (`non-neoplastic`, `benign`, `malignant`).
- **Data Partitioning:** 75% Training (171 images) | 25% Testing (57 images).
- **Backbone Architecture:** Pre-trained `MobileNetV3-Large` (ImageNet initial weights, 5.4M parameters).
- **Hardware Environment:** Executed on Nvidia Tesla T4 GPU (Google Colab).

---

## 📊 Empirical Findings & Prototype Validation

### 1. The Baseline Limitation: "The Accuracy Paradox"
When training a standard baseline model with standard Cross-Entropy Loss, evaluation on 57 unseen test images revealed a deceptive artifact:

| Demographic Cohort | Test Samples Evaluated | Observed Nominal Accuracy | Diagnostic Reality / Failure Mode |
| :--- | :---: | :---: | :--- |
| **Light Skin (Types I–III)** | 17 images | **64.71%** (11 / 17) | Balanced error distribution across classes. |
| **Dark / Indian Skin (Types IV–VI)** | 40 images | **87.50%** (35 / 40) | **Severe Majority-Class Shortcut (Overfitting).** |

- **Deep Confusion Matrix Inspection:** Out of 40 dark skin test images, 37 samples belonged strictly to the `non-neoplastic` (inflammatory) category. 
- The standard model learned a lazy frequency shortcut: it predicted `non-neoplastic` for almost every dark-skin image. While this yielded an artificially inflated 87.50% nominal accuracy, the model **completely failed (0% sensitivity) on minority `benign` and `malignant` lesions** (`existing_model_bias_proof.png`).

---

### 2. Proposed DermaEquity Intervention (Before vs. After)
To fix this limitation, our prototype introduced two core algorithmic interventions:
1. **Physics-Grounded Color Constancy (Shades of Gray):** Minkowski $p$-norm ($p=6$) illumination estimation to neutralize ambient room lighting variations and camera sensor shifts.
2. **Class-Balanced Fair Loss:** Dynamically assigning higher penalty weights to underrepresented minority classes to force the network to learn discriminative features rather than frequency shortcuts.

- **Outcome (`dermaequity_before_after_proof.png`):** As demonstrated in the side-by-side confusion matrix comparison, DermaEquity successfully broke out of the majority-class trap, establishing balanced multi-class recognition on darker skin cohorts.

---

### 3. Objective 2 Visual Explainability Validation (Grad-CAM++)
To verify that the model is making decisions based on true pathology rather than racial skin pigmentation, we conducted a visual explainability audit on dark skin clinical images (e.g., `lichen planus` on the upper lip, `gradcam_explainability_proof.png`):

- **Baseline Model Grad-CAM++ (The Flaw):** Saliency gradients exhibited severe **melanin background leakage**, dispersing heatmaps across facial hair, teeth, and normal pigmented skin rather than isolating the lesion.
- **DermaEquity Grad-CAM++ (The Fix):** With illumination normalization and balanced loss, the saliency gradient concentrated tightly on the active clinical lesion, confirming that visual attention is driven by pathological tissue morphology rather than background melanin.

---

## 🛠️ How to Replicate This Prototype

1. Open `model_py.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Enable GPU: `Runtime` > `Change runtime type` > `T4 GPU`.
3. Run all cells sequentially. The notebook will:
   - Ingest sample data and verify live links.
   - Train the baseline model and plot the initial bias audit.
   - Train the proposed DermaEquity pipeline with Shades of Gray color constancy.
   - Generate the before/after confusion matrices and 3-panel Grad-CAM++ explainability heatmaps.

---

## 🚀 Transition from Prototype to Full-Scale Training
With the end-to-end pipeline, loss formulation, and XAI safeguards empirically verified on this pilot prototype, the upcoming phase will scale up execution:
- [ ] Scaling dataset size to 3,000–5,000+ clinical and smartphone images from [Google SCIN](https://github.com/google-research-datasets/scin) and full Fitzpatrick17k archives.
- [ ] Extended fine-tuning across 25–30 epochs with learning rate scheduling and cross-validation.
- [ ] Computing formal quantitative metrics: Demographic Parity Difference, Equalized Odds ratio, and Energy-inside-BBox localization percentages.
