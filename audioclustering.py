import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from sklearn.cluster import DBSCAN

# ----------------------------------------------------------------------------------------------------
# Load the audio samples from the audio folder
# ----------------------------------------------------------------------------------------------------

audio_samples = "audio"
audio_files = os.listdir(audio_samples)


# Extracting features/characteristics from audios
spectrograms = []
names = []

for file in audio_files:
    path = os.path.join(audio_samples, file)
    # load the audio
    y, sr = librosa.load(path, sr=None) # load audio as a waveform y store the sampling rate as sr
    # compute mel spectrogram (learned this from the librosa documentation)
    # we do this because that's how human ear perceives sound
    S = librosa.feature.melspectrogram(y = y, sr = sr, n_mels = 64, fmax = 8000)
    # convert into decibels since they represent sound intensity more naturally and it's easier to see patterns
    S_db = librosa.power_to_db(S, ref=np.max) # convert to decibal scale
    spectrograms.append(S_db)
    names.append(file)

print("Spectrograms extracted successfully")
# now I need to compare spectrograms. since the spectrograms can be different size because the audio files are
# different in size, I decided to crop them all to the smallest common shape
# compute pairwise distances using matrix norm

def matrix_distance(A, B, norm_type="fro"):
    # compute distance between two spectograms using Frobenius
    max_rows = max(A.shape[0], B.shape[0])
    max_cols = max(A.shape[1], B.shape[1])
    # decided to pad silence to the shorter audio clip rather than crop it, so I
    # created padded versions (pad with minimum value, which is silence silence)
    A_padded = np.full((max_rows, max_cols), A.min()) # create another patrix where everything equals A.min
    B_padded = np.full((max_rows, max_cols), B.min())

    A_padded[:A.shape[0], :A.shape[1]] = A
    B_padded[:B.shape[0], :B.shape[1]] = B

    # calculate difference using frobenius norm
    diff  = A_padded - B_padded
    return np.linalg.norm(diff, ord=norm_type)

# Build distance matrix
n = len(spectrograms)
D = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        if i < j:
            D[i,j] = matrix_distance(spectrograms[i], spectrograms[j])
            D[j,i] = D[i,j]

print("Finding good eps value")
# for each audio, find distance to closest neighbor
nearest_neighbor_dists = []
for i in range(n):
    # get all distances for this file, excluding itself
    dists = [D[i][j] for j in range(n) if i != j]
    nearest_neighbor_dists.append(min(dists))

nearest_neighbor_dists.sort()

print("Distances to nearest neighbor for each file:")
for i, (name, dist) in enumerate(zip(names, nearest_neighbor_dists)):
    print(f"  {name}: {dist:.2f}")

print(f"\nSuggested eps values to try:")
print(f"  tight ep: {nearest_neighbor_dists[2]:.2f}")
print(f"  moderate ep: {nearest_neighbor_dists[4]:.2f}")
print(f"  loose ep: {nearest_neighbor_dists[6]:.2f}")

# Apply DBSCAN clustering
# eps is the max distance for samples to be neighbors
# min_samples is how many neighbors are needed to form a cluster
# tried different ep values. I played around with the values I got and stopped on this one
db = DBSCAN(eps = 7600, min_samples = 2, metric = "precomputed")
labels = db.fit_predict(D)

# print results
print("===Clustering Results===")
for i in range(len(names)):
    if labels[i] == -1:
        print(f"{names[i]}: Noise/Outlier")
    else:
        print(f"{names[i]}: Cluster {labels[i]}")

# Visualize the distance matrix to see the patterns

plt.hist(D.flatten(), bins=30, edgecolor="black")
plt.title("Distribution of pairwise distances")
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.show()

# some visualization
import seaborn as sns
plt.figure(figsize=(10,10))
sns.heatmap(D, xticklabels = names, yticklabels = names,  cmap="viridis")
plt.title("Pairwise distance matrix with Frobenius norm")
plt.show()


for i, (S_db, name) in enumerate(zip(spectrograms, names)):
    plt.figure(figsize=(8, 4))
    librosa.display.specshow(S_db, sr = sr, x_axis='time', y_axis='mel', cmap='magma')
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Mel Spectrogram - {name}")
    plt.tight_layout()
    plt.show()

# visualize the clustering with MDS
from sklearn.manifold import MDS

mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
positions = mds.fit_transform(D)

plt.figure(figsize=(12, 8))
unique_labels = set(labels)

colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
for label, color in zip(unique_labels, colors):
    if label == -1:
        # Noise points in black
        color = 'black'
        marker = 'x'
        label_name = 'Noise'
    else:
        marker = 'o'
        label_name = f'Cluster {label}'

    mask = labels == label
    plt.scatter(positions[mask, 0], positions[mask, 1],
               c=[color], label=label_name, s=200, marker=marker, edgecolors='black', linewidths=2)

plt.xlabel('MDS Dimension 1')
plt.ylabel('MDS Dimension 2')
plt.title(f'Audio Clustering Visualization (eps={7600}, min_samples={2})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()