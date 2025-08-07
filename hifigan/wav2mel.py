import torchaudio
import numpy as np
import torch
from librosa.filters import mel as librosa_mel_fn

class AudioProcessor:
    def __init__(self, n_fft=1024, num_mels=80, sampling_rate=22050, hop_size=256, win_size=1024, fmin=0, fmax=8000, center=False):
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.sampling_rate = sampling_rate
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.center = center
        self.mel_basis = {}
        self.hann_window = {}
        self.MAX_WAV_VALUE = 32768.0

    def load_wav(self, wav):
        target_sr=self.sampling_rate
        speech, sample_rate = torchaudio.load(wav)
        # Average multi-channel audio to obtain single-channel audio
        speech = speech.mean(dim=0, keepdim=True)
        if sample_rate != target_sr:
            assert sample_rate > target_sr, f'wav sample rate {sample_rate} must be greater than {target_sr}'
            speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
        return speech

    def dynamic_range_compression(self, x, C=1, clip_val=1e-5):
        return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)

    def dynamic_range_decompression(self, x, C=1):
        return np.exp(x) / C

    def dynamic_range_compression_torch(self, x, C=1, clip_val=1e-5):
        return torch.log(torch.clamp(x, min=clip_val) * C)

    def dynamic_range_decompression_torch(self, x, C=1):
        return torch.exp(x) / C

    def spectral_normalize_torch(self, magnitudes):
        return self.dynamic_range_compression_torch(magnitudes)

    def spectral_de_normalize_torch(self, magnitudes):
        return self.dynamic_range_decompression_torch(magnitudes)

    def mel_spectrogram(self, y):
        if torch.min(y) < -1.0:
            print("min value is ", torch.min(y))
        if torch.max(y) > 1.0:
            print("max value is ", torch.max(y))

        device = y.device
        if f"{str(self.fmax)}_{str(device)}" not in self.mel_basis:
            mel = librosa_mel_fn(sr=self.sampling_rate, n_fft=self.n_fft, n_mels=self.num_mels, fmin=self.fmin, fmax=self.fmax)
            self.mel_basis[f"{self.fmax}_{device}"] = torch.from_numpy(mel).float().to(device)
            self.hann_window[str(device)] = torch.hann_window(self.win_size).to(device)

        y = torch.nn.functional.pad(
            y.unsqueeze(1), (int((self.n_fft - self.hop_size) / 2), int((self.n_fft - self.hop_size) / 2)), mode="reflect"
        ).squeeze(1)

        spec = torch.view_as_real(
            torch.stft(
                y,
                self.n_fft,
                hop_length=self.hop_size,
                win_length=self.win_size,
                window=self.hann_window[str(device)],
                center=self.center,
                pad_mode="reflect",
                normalized=False,
                onesided=True,
                return_complex=True,
            )
        )

        spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
        spec = torch.matmul(self.mel_basis[f"{self.fmax}_{device}"], spec)
        spec = self.spectral_normalize_torch(spec)

        return spec

    def extract_speech_feat(self, speech):
        speech_feat = self.mel_spectrogram(speech).squeeze(dim=0).transpose(0, 1)
        speech_feat = speech_feat.unsqueeze(dim=0)
        speech_feat_len = torch.tensor([speech_feat.shape[1]], dtype=torch.int32)
        return speech_feat, speech_feat_len

    def audio_to_mel(self, file_path):
        speech = self.load_wav(file_path)
        speech_feat, speech_feat_len = self.extract_speech_feat(speech)
        return speech, speech_feat, speech_feat_len
        
    def audio_array2file(audio_array, file_path):
        sf.write(out_wav_path,(audio_array * 32768).astype(np.int16),32000)

if __name__ == '__main__':
    # Example usage:
    file_path = '/home/qiangchunyu/data/test_data/wav_24k/000002.wav'
    audio_processor = AudioProcessor()
    speech, speech_feat, speech_feat_len = audio_processor.audio_to_mel(file_path)
    print('speech:', speech.shape)
    print('speech_feat:', speech_feat.shape)
    print('speech_feat_len:', speech_feat_len)

