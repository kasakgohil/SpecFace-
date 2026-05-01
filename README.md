# Capstone: FFT-Based Image Classification

This repository contains the source code for an image classification pipeline (STL-10 and VGGFace) utilizing novel Blockwise and Global Fast Fourier Transform (FFT) features combined with XGBoost and PyTorch neural networks.

## Abstract

Image classification is a huge part of modern computer vision, and deep learning models have gotten incredibly good at it. However, most standard Convolutional Neural Networks (CNNs) focus almost entirely on the spatial features of an image—the actual pixels. In doing so, they often miss out on the valuable, hidden patterns found in the frequency domain. This can hold back a model's overall robustness and make it harder to understand how it's making decisions, especially for tricky tasks like recognizing objects or analyzing faces.

To bridge this gap, our project introduces a new way to handle image classification by looking at both the spatial and frequency details. Instead of just looking at raw pixels like most conventional methods do, we use Fast Fourier Transform (FFT) techniques. By extracting both global and block-by-block frequency features, our approach gets a much deeper understanding of an image's textures and underlying structure before it even tries to classify it.

Our system uses a hybrid approach, combining traditional frequency analysis with modern deep neural networks. We start by cleaning up the images using tools like Histogram Equalization and Gaussian Blur. Then, we use our FFT techniques to pull out the key frequency features. We shrink these features down to their most important components using PCA, and feed them into powerful classifiers like XGBoost and PyTorch models (like ResNet18 and our own custom FFTNets). To make sure our models are actually looking at the right things—and not just memorizing random background noise—we use Grad-CAM to visually explain the model's choices.

We trained and thoroughly tested this framework on standard datasets like STL-10 and VGGFace. The whole pipeline is built to be modular, so it's easy to run on remote servers with multiple GPUs. Based on our testing, combining these unique frequency features with standard deep learning techniques gives us highly competitive results while making the models much easier to interpret.

Ultimately, looking at the frequency information of an image proves to be a powerful upgrade to standard pixel-based approaches. This project offers a scalable, effective, and transparent way to push image classification forward.

## Architecture

The codebase has been refactored for modularity and remote execution:
- `configs/`: Centralized `.yaml` configurations.
- `src/`: Core logic including models (`FFTNet`, `ResNet18`), data preprocessing (Histogram EQ, Gaussian Blur, FFT extraction), and training loops.
- `scripts/`: Entry points for training different models.
- `artifacts/`: Automatically generated directory to store models (`.pth`, `.json`), logs (`.csv`), and scalers (`.pkl`).

## Remote Server / SSH Setup

### 1. Requirements
- Python 3.8+
- NVIDIA GPU (Recommended) with CUDA installed.

### 2. Configuration
Update the paths inside `configs/default_config.yaml` to point to your raw binaries or dataset directories before training.

### 3. Execution
You can use the interactive bash script to setup the environment and start training automatically:
```bash
chmod +x run_server.sh
./run_server.sh
```

Alternatively, you can manually run the scripts:
```bash
# 1. Setup Data (Extract binaries, Apply HistEq + Denoise)
python scripts/setup_data.py

# 2. Train ResNet Baseline
python scripts/train_resnet.py

# 3. Train Global FFT models (XGBoost & FFTGlobalNet)
python scripts/train_fft_global.py

# 4. Train Blockwise FFT models (Handcrafted features)
python scripts/train_fft_blockwise.py

# 5. Train Blockwise FFT models with PCA
python scripts/train_fft_blockwise.py --use-pca

# 6. Train Grad-CAM Enriched Blockwise FFT
python scripts/train_fft_blockwise.py --use-gradcam
```

## Logs & Artifacts
All trained models and logs are automatically saved to the `artifacts/` folder. Training metrics (accuracy, loss) will print to standard output (suitable for `nohup` or `tmux` sessions) and save as CSVs in `artifacts/logs/`.
