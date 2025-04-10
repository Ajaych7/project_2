import os
import librosa
import numpy as np
import soundfile as sf
from sklearn.model_selection import train_test_split

def load_audio_files(directory, sample_rate=22050):
    """Load audio files from directory and return spectrograms"""
    audio_data = []
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            filepath = os.path.join(directory, filename)
            audio, sr = librosa.load(filepath, sr=sample_rate)
            # Convert to mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
            audio_data.append(mel_spec)
    return np.array(audio_data)

# Example usage
audio_dir = "path/to/your/audio/files"
spectrograms = load_audio_files(audio_dir)

# Split data into training and validation sets
X_train, X_val = train_test_split(spectrograms, test_size=0.2, random_state=42)
