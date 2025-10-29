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

spectrograms = []
names = []

for file in audio_files:
    path = os.path.join(audio_samples, file)
    y, sr = librosa.load(path, sr=None) # load audio as a waveform y store the sampling rate as sr
    # compute mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=8000)
    S_db = librosa.power_to_db(S, ref=np.max) # convert to decibal scale
    spectrograms.append(S_db)
    names.append(file)

print(f"Loaded {len(spectrograms)} spectograms.")

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
n = len(spectrograms)
D = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        if i < j:
            D[i,j] = matrix_distance(spectrograms[i], spectrograms[j])
            D[j,i] = D[i,j]

# Normalize distances for dbscan
# D = D / np.max(D)

# cluster using dbscan
db = DBSCAN(eps = 1000, min_samples = 2, metric = "precomputed")
labels = db.fit_predict(D)

# print results
print("===Clustering Results===")
for name, label in zip(names, labels):
    print(f"{name} : Cluster {label}")

plt.hist(D.flatten(), bins=30)
plt.title("Distribution of pairwise distances")
plt.show()

# some visualization
import seaborn as sns
plt.figure(figsize=(10,10))
sns.heatmap(D, xticklabels = names, yticklabels = names,  cmap="viridis")
plt.title("Pairwise distance matrix with Frobenius norm")
plt.show()


for i, (S_db, name) in enumerate(zip(spectrograms, names)):
    plt.figure(figsize=(8, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', cmap='magma')
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Mel Spectrogram - {name}")
    plt.tight_layout()
    plt.show()
