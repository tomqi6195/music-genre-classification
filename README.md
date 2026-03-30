# 🎶 Deep Audio Genre Classifier

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)
![Gradio](https://img.shields.io/badge/Gradio-UI-ff7c00.svg)

An end-to-end deep learning project that classifies music genres from raw audio files. This project explores the GTZAN dataset, moving from baseline Machine Learning models (SVM) to custom Neural Networks, Transfer Learning, and ultimately proving that **Soft-Voting Ensembles outperform complex Meta-Classifiers** on subjective audio data.

The project concludes with a fully offline, interactive Gradio Web App that slices 30-second audio tracks into 3-second rolling windows for highly stabilized inference.

*(Insert a screenshot of your beautiful Gradio UI here!)*
---

## 🧠 Model Architectures & Pipeline

To capture the complex nature of music, this project utilizes a dual-expert ensemble approach:

1. **The Rhythm Expert (Custom Multi-Scale CNN):** Built from scratch using asymmetric convolutions (1x7 and 7x1) and Squeeze-and-Excitation (SE) blocks to explicitly isolate temporal drum beats and rhythmic transients.
2. **The Texture Expert (ResNet-50):** Fine-tuned using ImageNet transfer learning. It analyzes 128x130 Mel Spectrograms as dense 2D images to capture deep harmonic clouds and timbral textures.
3. **The Ultimate Ensemble:** A Soft-Voting averager that smoothly blends the probabilistic outputs of both models.

**Why not a Meta-Classifier?** Extensive experiments were run using Logistic Regression and Random Forest Meta-Classifiers. The findings proved that on a dataset of this size (1,000 tracks), complex stacked classifiers aggressively overfit the validation set. A simple mathematical Soft-Ensemble or a hardcoded Heuristic Router yielded the highest, most robust F1-scores.

---

## 📊 Final Results (Strict 10% Test Set)

| Model Setup | Test Accuracy | Notable Strengths |
| :--- | :--- | :--- |
| Baseline SVM | ~50% | None |
| Custom Multi-Scale CNN | 81% | Metal (92% F1), Reggae |
| ResNet-50 (Transfer Learning) | 82% | Rock, Classical, Jazz |
| Random Forest Meta-Classifier | 80% | *Failed due to overfitting* |
| **Soft-Voting Ensemble** | **85%** | **Highly balanced across all classes** |

---

## 🚀 How to Reproduce & Run Locally

### 1. Clone the Repository
```bash
git clone [https://github.com/tomqi6195/music-genre-classification.git](https://github.com/tomqi6195/music-genre-classification.git)
cd music-genre-classification
