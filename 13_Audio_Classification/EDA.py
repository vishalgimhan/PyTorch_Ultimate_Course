# %% Packages
import torchaudio
torchaudio.set_audio_backend("soundfile")
from plot_audio import plot_specgram, plot_waveform
import seaborn as sns
import matplotlib.pyplot as plt

# %% Check if audio backend installed
# pip install soundfile
# pip install ffmpeg

# %% Import data
wav_file = "13_data/set_a/extrahls__201101070953.wav"
data_waveform, sr = torchaudio.load(wav_file)
data_waveform.size()

# %% Plot Waveform
# plot_waveform(data_waveform, sample_rate=sr)

# %% Calculate Spectrogram
spectogram = torchaudio.transforms.Spectrogram()(data_waveform)
spectogram.size()

# %% Plot Spectrogram
# plot_specgram(waveform=data_waveform, sample_rate=sr)

# %%
