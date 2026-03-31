# Deep Audio Genre Classifier

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)
![Gradio](https://img.shields.io/badge/Gradio-UI-ff7c00.svg)

This project classifies raw audio files into 10 different music genres. It uses the GTZAN dataset for training, exploring different architectures including the baseline support vector machine (SVM), custom neural networks, transfer learning, and ensembles.

It also has a fully offline, interactive Gradio web app that performs inference on 30-second audio tracks in MP3 or WAV format.

<img width="2846" height="1460" alt="image" src="https://github.com/user-attachments/assets/c65c0677-b5b7-49cb-bda8-9291e501432d" />

---

## Model Architectures & Pipeline

To capture the complex nature of music, this project utilizes a dual-expert ensemble approach:

1. **The Rhythm Expert (SpectroCNN):** Built from scratch using asymmetric convolutions and squeeze-and-excitation blocks to explicitly isolate temporal drum beats and rhythmic transients.
2. **The Texture Expert (ResNet-50):** Fine-tuned using ImageNet transfer learning, which analyzes 128x130 Mel Spectrograms as dense 2D images to capture deep harmonic clouds and timbral textures.
3. **The Ultimate Ensemble:** A soft-voting averager that smoothly blends the probabilistic outputs of both models.

---

## Final Results (10% Test Set)

| Model Setup | Precision | Recall | F1-Score | Notable Strengths |
| :--- | :--- | :--- | :--- | :--- |
| Baseline SVM | 0.71 | 0.70 | 0.70 | None |
| SpectroCNN | 0.83 | 0.81 | 0.82 | Blues, Classical, Metal |
| ResNet-50 | 0.83 | 0.82 | 0.83 | Disco, Hiphop, Jazz, Metal |
| **Soft-Voting Ensemble** | **0.86** | **0.84** | **0.85** | **Highly balanced across classes (except Rock)** |

---

## How to Reproduce & Run Locally

### 1. Clone the Repository
```bash
git clone [https://github.com/tomqi6195/music-genre-classification.git](https://github.com/tomqi6195/music-genre-classification.git)
cd music-genre-classification
```

### 2. Install Dependencies
This project uses librosa to safely decode MP3/WAV files without requiring system-level FFmpeg installations.
```bash
pip install torch torchvision torchaudio librosa gradio matplotlib pillow numpy
```



---

## Repository Structure
- The ```architecture``` folder contains all the Colab notebooks used the construct models, in which ```EnsembleExperiments.ipynb``` is the master notebook containing the strict data-splitting pipeline, model class definitions, the training Loop, and the comprehensive evaluation of all ensemble techniques.
- The ```demo``` folder includes files ```app.py``` (the Gradio web app script) as well as ```best_spectro_cnn.pth``` (the optimized weights for the rhythm-detecting SpectroCNN).
- The ```data_processing``` folder contains all the preliminary data processing code, which are then adapted and used in the architecture notebooks.
