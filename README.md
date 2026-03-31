# Music Genre Classifier

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)
![Gradio](https://img.shields.io/badge/Gradio-UI-ff7c00.svg)

This project classifies raw audio files into 10 different music genres. It used the GTZAN - Music Genre Classification dataset for training, and explored different architectures including the baseline support vector machine (SVM), custom neural networks, transfer learning, and ensembles.

It also has a fully offline, interactive Gradio web app that performs inference on 30-second audio tracks in MP3 or WAV format.

<img width="2846" height="1460" alt="image" src="https://github.com/user-attachments/assets/c65c0677-b5b7-49cb-bda8-9291e501432d" />

---

## Data Augmentation
During training, the audio data was augmented using 3 different techniques:
1. **Slicing:** Each 30-second WAV file was sliced into ten 3-second segments, increasing the size of the dataset tenfold.
2. **Pitch-Shifting:** Each sample had a 50% chance of being pitch-shifted by 2 semitones to generalize the model to pitch differences.
3. **White Noise Addition:** Random white noise was added onto the samples with a 50% chance to make the model robust to background noise and real-world imperfections. 

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
| SpectroCNN | 0.83 | 0.81 | 0.82 | Blues, Classical, Metal, Pop |
| ResNet-50 | 0.83 | 0.82 | 0.83 | Country, Disco, Hiphop, Jazz |
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

### 3. Run the Interactive Web App
The pre-trained weights for the SpectroCNN (```best_spectro_cnn.pth```) are included in the repository, but the file containing the fine-tuned weights for ResNet-50 is too large to be uploaded. Use the file ```resnet.ipynb``` in the architecture folder, and manually obtain the ResNet weights in Google Colab (```best_resnet50.pth```). Ensure both files are in the root directory, then run:
```bash
python app.py
```
A local link will appear in the terminal. Click it to open the UI, upload an MP3 or WAV file, and watch the ensemble model generate a 30-second Mel Spectrogram and predict the genre!

---

## Repository Structure
- The ```architecture``` folder contains all the Colab notebooks used the construct models, in which ```EnsembleExperiments.ipynb``` is the master notebook containing the strict data-splitting pipeline, model class definitions, the training Loop, and the comprehensive evaluation of all ensemble techniques.
- The ```data``` folder includes data processing code, which is adapted and used in the architecture notebooks.
- The ```demo``` folder contains files ```app.py``` (the Gradio web app script) as well as ```best_spectro_cnn.pth``` (the optimized weights for the rhythm-detecting SpectroCNN).
