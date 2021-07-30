# Learning note-to-note affinity for voice segregation and melody line identification of symbolic music data
This repository contains the implementation of the data-driven system to solve the voice segregation problem for symbolic music.
You can read the full paper presented at ISMIR 2021 for further details.

## Abstract
Voice segregation, melody line identification and other tasks of identifying the horizontal elements of music have been developed independently, although their purposes are similar. 
In this paper, we propose a unified framework to solve the voice segregation and melody line identification tasks of symbolic music data. 
To achieve this, a neural network model is trained to learn note-to-note affinity values directly from their contextual notes, in order to represent a music piece as a weighted undirected graph, with the affinity values being the edge weights. 
Individual voices or streams are then obtained with spectral clustering over the learned graph. 
Conditioned on minimal prior knowledge, the framework can achieve state-of-the-art performance on both tasks, and further demonstrates strong advantages on simulated real-world symbolic music data with missing notes and asynchronous chord notes.
