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

def matrix_distance(A,B, norm_type="fro"):
    # compute distance between two spectograms using Frobenius
    # resize the smallest common shape
    min_rows = min(A.shape[0], B.shape[0])
    min_cols = min(A.shape[1], B.shape[1])

    A_cut = A[:min_rows, :min_cols]
    B_cut = B[:min_rows, :min_cols]
    diff = A_cut - B_cut
    return np.linalg.norm(diff, ord=norm_type)

# Build distance matrix
n = len(spectograms)
D = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        if i < j:
            D[i,j] = matrix_distance(spectograms[i], spectograms[j])
            D[j,i] = D[i,j]

# Normalize distances for dbscan
D = D / np.max(D)

# cluster using dbscan