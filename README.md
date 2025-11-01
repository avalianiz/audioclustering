# 🎵 Audio Clustering using DBSCAN
The goal I had in mind was to compare and cluster a set of audio samples obtained from [here](https://freesound.org/) based on their acoustic similarity measured 
from their **Mel spectrogram representations**. Each audio file was converted into a 2D matrix (a spectrogram) showing how the energy is distributed across frequencies 
over the entire sound interval. Spectrograms preserve both spectral and temporal features, which give us the perfect way of understanding the differences and relations 
between audio files. This model relies on the DBSCAN algorithm.
