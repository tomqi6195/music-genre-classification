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

| Model Setup | Test Accuracy (F1-Score) | Notable Strengths |
| :--- | :--- | :--- |
| Baseline SVM | 70% | None |
| SpectroCNN | 82% | Blues, Classical, Metal |
| ResNet-50 (Transfer Learning) | 83% | Disco, Hiphop, Jazz, Metal |
| **Soft-Voting Ensemble** | **85%** | **Highly balanced across all classes (except Rock)** |

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

### 3. Run the Interactive Web App
The pre-trained weights for the SpectroCNN (best_spectro_cnn.pth) are included in the repository, but the file containing the fine-tuned weights for ResNet-50 is too large to be uploaded. Use the file ```resnet.ipynb``` in the architecture folder, and manually obtain the ResNet weights in Google Colab (best_resnet50.pth). Ensure both files are in the root directory, then run:
```bash
python app.py
```
A local link will appear in the terminal. Click it to open the UI, upload an MP3 or WAV file, and watch the ensemble model generate a 30-second Mel Spectrogram and predict the genre!

## Repository Structure
- The ```architecture``` folder contains all the Colab notebooks used the construct models, in which ```EnsembleExperiments.ipynb``` is the master notebook containing the strict data-splitting pipeline, model class definitions, the training Loop, and the comprehensive evaluation of all ensemble techniques.
- The ```demo``` folder has ```app.py``` (the Gradio web app script) as well as ```best_spectro_cnn.pth``` (the optimized weights for the rhythm-detecting SpectroCNN).
- The ```data_processing``` folder includes all the preliminary data processing notebooks, which are then adapted and used in the architecture notebooks.
