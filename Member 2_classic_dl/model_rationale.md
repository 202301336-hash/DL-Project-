# Classic Deep Learning: Model-Selection Rationale

## 1. Comparison of Classic Deep Learning Architectures for Time-Frequency Audio
When processing environmental audio represented as 2D time-frequency spectrograms (**Log-Mel Spectrograms**), classical deep learning architectures perform differently based on how they handle temporal and spectral structures:

| Architecture | Suitability for Time-Frequency Audio | Strengths & Limitations on ESC-50 (2,000 samples) |
| :--- | :--- | :--- |
| **MLP (Multi-Layer Perceptron)** | **Low** | Flattening the spectrogram destroys 2D time-frequency locality. High parameter count leads to severe overfitting on the 1,280 training samples. |
| **1D-CNN** | **Moderate** | Captures temporal patterns along the time axis, but treats frequency bins as independent channels, failing to exploit local spectral correlations. |
| **RNN / LSTM / GRU** | **Moderate to Good** | Effective at modeling long-range temporal dependencies, but computationally slower and less effective at extracting local spectral features on its own. |
| **2D-CNN (Chosen Model)** | **High (Optimal)** | Treats the Log-Mel spectrogram as a 2D image. Convolutional kernels simultaneously capture local spectral patterns (harmonics/pitch) and temporal transitions, while weight sharing prevents overfitting. |
| **CRNN (2D-CNN + BiGRU)** | **High (Alternative)** | Combines 2D spatial feature extraction with recurrent temporal modeling, though a regularized 2D-CNN converges faster and generalizes more reliably on small datasets like ESC-50. |

## 2. Justification of the Chosen Architecture (2D-CNN)
We selected a **4-Block 2D Convolutional Neural Network (2D-CNN)** equipped with **Batch Normalization**, **Dropout**, and **Global Average Pooling** for the following reasons:
1. **Time-Frequency Locality:** Environmental sounds in ESC-50 exhibit distinct spectro-temporal signatures across adjacent frequency bands and time frames, which 2D convolutional filters extract efficiently.
2. **Shift Invariance:** `MaxPooling2d` layers allow the network to recognize sound events regardless of where they occur within the 5-second audio clip.
3. **Overfitting Prevention on Small Data:** With a dataset split of **1,280 training (64%)**, **320 validation (16%)**, and **400 test (20%)** samples, replacing large fully-connected layers with `AdaptiveAvgPool2d((1, 1))` drastically reduces parameter count. Combined with **SpecAugment** and **AWGN Noise Injection**, the model achieves strong generalization and noise robustness.