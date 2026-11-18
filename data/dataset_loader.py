import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torchaudio
from scipy.io import wavfile

class ESC50SpectrogramDataset(Dataset):
    """
    Custom Dataset loader for ESC-50 Spectrograms (Member 4 - Advanced Architecture)
    """
    def __init__(self, metadata_df, base_data_dir, transform=None):
        self.metadata = metadata_df.reset_index(drop=True)
        self.base_data_dir = base_data_dir
        self.transform = transform
        
        # تحويل الصوت إلى Mel-Spectrogram
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=22050,
            n_fft=1024,
            hop_length=512,
            n_mels=128
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        filename = str(row['filename'])
        target = int(row['target'])
        
        audio_path = os.path.join(self.base_data_dir, filename)
        if not os.path.exists(audio_path):
            audio_path = os.path.join(os.path.dirname(self.base_data_dir), "audio", filename)

        if os.path.exists(audio_path):
            # قراءة الملف عبر scipy لتفادي مشكلة torchcodec
            sr, data = wavfile.read(audio_path)
            
            # تحويل البيانات إلى float32 وتطبيع القيم بين [-1.0, 1.0]
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                data = data.astype(np.float32) / 2147483648.0
            else:
                data = data.astype(np.float32)

            waveform = torch.from_numpy(data)
            
            # التأكد من أبعاد الـ Waveform [Channels, Time]
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            elif waveform.dim() == 2:
                waveform = waveform.t()  # تحويل [Time, Channels] إلى [Channels, Time]
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # تحويل إلى Mel-Spectrogram
            spectrogram = self.amplitude_to_db(self.mel_spectrogram(waveform))
            
            # تطبيع السبيكتروجرام (Normalization)
            spectrogram = (spectrogram - spectrogram.mean()) / (spectrogram.std() + 1e-6)
        else:
            raise FileNotFoundError(f"❌ لم يتم العثور على الملف الصوتي: {audio_path}")

        if spectrogram.dim() == 2:
            spectrogram = spectrogram.unsqueeze(0)

        if self.transform:
            spectrogram = self.transform(spectrogram)

        return spectrogram, torch.tensor(target, dtype=torch.long)