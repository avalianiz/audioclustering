import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform


# --------- load the audio samples

audio_samples = "audio"
audio_files = os.listdir(audio_samples) # get the audios from the folder

spectograms = []
names = []

for file in audio_files:
    path = os.path.join(audio_samples, file)
    y, sr = librosa.load(path, sr=None) # load audio as a waveform y store the sampling rate as sr
    # compute mel spectogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=8000)
    S_db = librosa.power_to_db(S, ref=np.max) # convert to decibal scale
    spectograms.append(S_db)
    names.append(file)

print(f"Loaded {len(spectograms)} spectograms.")

# compute pairwise distances using matrix norm
