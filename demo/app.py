import io
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt
from PIL import Image
import gradio as gr
import librosa

# ==========================================
# 1. ARCHITECTURES (Needed to load the weights)
# ==========================================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.excitation(self.squeeze(x).view(b, c)).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MultiScaleSpectroBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleSpectroBlock, self).__init__()
        self.conv_std = nn.Conv2d(in_channels, out_channels // 3, kernel_size=3, padding='same')
        self.conv_tmp = nn.Conv2d(in_channels, out_channels // 3, kernel_size=(1, 7), padding='same')
        self.conv_frq = nn.Conv2d(in_channels, out_channels - 2*(out_channels // 3), kernel_size=(7, 1), padding='same')
        self.bn = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        out = torch.cat([self.conv_std(x), self.conv_tmp(x), self.conv_frq(x)], dim=1)
        return self.pool(self.se(F.leaky_relu(self.bn(out), 0.01)))

class CreativeSpectroCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CreativeSpectroCNN, self).__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.01)
        )
        self.blocks = nn.Sequential(
            MultiScaleSpectroBlock(32, 64),
            MultiScaleSpectroBlock(64, 128),
            MultiScaleSpectroBlock(128, 256),
            MultiScaleSpectroBlock(256, 512)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.01),
            nn.Dropout(0.5), nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.pool(self.blocks(self.init_conv(x))).view(x.size(0), -1)
        return self.classifier(x)

class SpectroResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super(SpectroResNet50, self).__init__()
        self.resnet = models.resnet50()
        orig_weight = self.resnet.conv1.weight.clone()
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.resnet.conv1.weight = nn.Parameter(torch.sum(orig_weight, dim=1, keepdim=True))
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, num_classes))
    def forward(self, x):
        return self.resnet(x)

# ==========================================
# 2. LOAD MODELS 
# ==========================================
device = torch.device("cpu")

demo_cnn = CreativeSpectroCNN(num_classes=10).to(device)
demo_cnn.load_state_dict(torch.load('best_spectro_cnn.pth', map_location=device))
demo_cnn.eval()

demo_resnet = SpectroResNet50(num_classes=10).to(device)
demo_resnet.load_state_dict(torch.load('best_resnet50.pth', map_location=device))
demo_resnet.eval()

genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# ==========================================
# 3. PREDICTION LOGIC (LIBROSA + 30s CHUNKING)
# ==========================================
def predict_and_visualize(audio_path):
    if audio_path is None:
        return None, "Please upload an audio file."
    
    target_sr = 22050
    n_samples_30s = target_sr * 30 
    
    try:
        y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    except Exception as e:
        return None, f"Error loading audio: {str(e)}"
    
    waveform = torch.tensor(y).unsqueeze(0) 
        
    if waveform.shape[1] < n_samples_30s:
        waveform = F.pad(waveform, (0, n_samples_30s - waveform.shape[1]))
    else:
        waveform = waveform[:, :n_samples_30s]
        
    mel_spec_full = torchaudio.transforms.AmplitudeToDB(top_db=80)(
        torchaudio.transforms.MelSpectrogram(sample_rate=target_sr, n_fft=2048, hop_length=512, n_mels=128)(waveform)
    )
    
    fig, ax = plt.subplots(figsize=(10, 4)) 
    # Switched to 'inferno' colormap for a more fiery/neon look to match the dark theme!
    ax.imshow(mel_spec_full.squeeze().numpy(), cmap='inferno', origin='lower', aspect='auto')
    ax.axis('off')
    fig.tight_layout(pad=0)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor='#0f172a') # Match background
    buf.seek(0)
    spectro_img = Image.open(buf)
    plt.close(fig)
    
    cnn_probs = []
    resnet_probs = []
    
    frames_per_chunk = 130 # This is EXACTLY 3 seconds!
    
    for i in range(10): # 10 chunks * 3 seconds = 30 seconds total
        start_frame = i * frames_per_chunk
        end_frame = start_frame + frames_per_chunk
        
        mel_chunk = mel_spec_full[:, :, start_frame:end_frame]
        
        if mel_chunk.shape[2] < frames_per_chunk:
            mel_chunk = F.pad(mel_chunk, (0, frames_per_chunk - mel_chunk.shape[2]))
            
        mel_chunk_tensor = mel_chunk.unsqueeze(0).to(device)
        
        with torch.no_grad():
            prob_cnn = F.softmax(demo_cnn(mel_chunk_tensor), dim=1)
            prob_resnet = F.softmax(demo_resnet(mel_chunk_tensor), dim=1)
            cnn_probs.append(prob_cnn)
            resnet_probs.append(prob_resnet)
            
    avg_cnn = torch.cat(cnn_probs).mean(dim=0)
    avg_resnet = torch.cat(resnet_probs).mean(dim=0)
    
    final_prob = ((avg_cnn + avg_resnet) / 2.0).cpu().numpy()
        
    confidences = {genres[i]: float(final_prob[i]) for i in range(len(genres))}
    return spectro_img, confidences

# ==========================================
# 4. CUSTOM GRADIO UI (THE MUSIC THEME)
# ==========================================
# Creating a custom sleek Dark Mode theme
music_theme = gr.themes.Default(
    primary_hue="fuchsia",     # Neon pinkish-purple accents
    secondary_hue="cyan",      # Bright blue highlights
    neutral_hue="slate",       # Deep dark blues/grays for the background
    font=[gr.themes.GoogleFont("Outfit"), "sans-serif"] # Modern, geometric font
).set(
    body_background_fill="#0f172a",          # Very dark slate background
    body_background_fill_dark="#0f172a",
    block_background_fill="#1e293b",         # Slightly lighter slate for the panels
    block_background_fill_dark="#1e293b",
    body_text_color="#f8fafc",               # Crisp white text
    body_text_color_dark="#f8fafc",
    button_primary_background_fill="#d946ef", # Neon Fuchsia button
    button_primary_background_fill_dark="#d946ef",
    button_primary_text_color="#ffffff",
    slider_color="#22d3ee",                  # Neon Cyan accents
)

with gr.Blocks() as interface:
    gr.Markdown("<h1 style='text-align: center; color: #22d3ee;'>Deep Audio Genre Classifier</h1>")
    gr.Markdown("<p style='text-align: center;'>Upload an MP3/WAV! The Soft-Voting Ensemble will slice the first 30 seconds into 3-second chunks and analyze the acoustic textures.</p>")
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(type="filepath", label="Upload Song")
            submit_btn = gr.Button("Analyze Track", variant="primary")
        
        with gr.Column(scale=2):
            image_output = gr.Image(label="Mel Spectrogram (30s Window)", type="pil")
            label_output = gr.Label(num_top_classes=4, label="Ensemble Prediction")
            
    submit_btn.click(fn=predict_and_visualize, inputs=audio_input, outputs=[image_output, label_output])

if __name__ == "__main__":
    interface.launch(share=False, theme=music_theme)
