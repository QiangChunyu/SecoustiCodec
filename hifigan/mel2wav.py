import os
import sys
import torch
import torchaudio
import numpy as np
from .generator import HiFTGenerator
from .wav2mel import AudioProcessor
# Set environment variable for CUDA
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def initialize_device():
    """Initialize the device for PyTorch."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_path, device):
    """Load the HiFTGenerator model from the specified path."""
    model = HiFTGenerator().to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    return model

def process_audio(file_path):
    """Process the audio file and return speech, features, and feature lengths."""
    audio_processor = AudioProcessor()
    speech, speech_feat, speech_feat_len = audio_processor.audio_to_mel(file_path)
    return speech, speech_feat, speech_feat_len

def infer_tts(mel, hift_path):
    device = torch.device('cpu')  # 将设备设置为CPU
    mel = mel.to(device)  # 将输入数据移动到CPU
    hift = load_model(hift_path, device)  # 加载模型并移动到CPU
    hift = hift.to(device)  # 将模型移动到CPU
    with torch.no_grad():
        """Generate TTS speech from mel spectrogram."""
        return hift.inference(mel=torch.swapaxes(mel, 1, 2))

def infer_tts_device(mel, device):
    print('hifigan:', device)
    mel = mel.to(device)  # 将输入数据移动到CPU
    hift_path = './pretrain_model/hift.pt'
    hift = load_model(hift_path, device)  # 加载模型并移动到CPU
    hift = hift.to(device)  # 将模型移动到CPU
    with torch.no_grad():
        """Generate TTS speech from mel spectrogram."""
        return hift.inference(mel=torch.swapaxes(mel, 1, 2))

def save_audio(file_name, audio_tensor, sample_rate=22050):
    """Save the generated audio to a file."""
    torchaudio.save(file_name, audio_tensor, sample_rate)