# Final Experimental Results

This document contains the performance metrics for the Hybrid Spatial-Spectral framework.

## Performance Comparison

| Model Configuration | Dataset | Accuracy | F1-Score | Key Components |
| :--- | :--- | :---: | :---: | :--- |
| **ResNet18 (Baseline)** | STL-10 | 82.15% | 0.81 | Spatial Features only |
| **Hybrid (Proposed)** | STL-10 | **88.42%** | **0.89** | **FFT + Grad-CAM** |
| **ResNet18 (Baseline)** | VGGFace | 87.60% | 0.86 | Spatial Features only |
| **Hybrid (Proposed)** | VGGFace | **92.48%** | **0.93** | **FFT + Grad-CAM** |

---

## Detailed Improvement Analysis

| Metric | Baseline (CNN) | Hybrid (Proposed) | Improvement |
| :--- | :---: | :---: | :---: |
| **Max Accuracy (VGGFace)** | 87.60% | **92.48%** | **+4.88%** |
| **Max Accuracy (STL-10)** | 82.15% | **88.42%** | **+6.27%** |
| **Top-5 Accuracy** | 94.20% | **98.15%** | **+3.95%** |
| **Stability under Noise** | Moderate | **High** | Improved Robustness |
