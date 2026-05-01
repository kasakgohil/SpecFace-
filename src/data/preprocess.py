import numpy as np
import cv2
import torch

def apply_histeq(images):
    result = []
    for img in images:
        img_uint8 = (img * 255).astype(np.uint8)
        eq = cv2.equalizeHist(img_uint8)
        result.append(eq / 255.0)
    return np.array(result)

def apply_denoise(images, ksize=3):
    result = []
    for img in images:
        img_uint8 = (img * 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(img_uint8, (ksize, ksize), 0)
        result.append(blurred / 255.0)
    return np.array(result)

def extract_global_fft(image, img_size=96, device="cpu"):
    image = cv2.resize(image, (img_size, img_size))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    tensor = torch.tensor(gray, dtype=torch.float32, device=device)

    fft = torch.fft.fft2(tensor)
    fft_shift = torch.fft.fftshift(fft)

    magnitude = torch.abs(fft_shift)
    magnitude = torch.log1p(magnitude)

    magnitude = magnitude / (torch.max(magnitude) + 1e-8)

    return magnitude.flatten().cpu().numpy()

def extract_blockwise_fft(image, img_size=96, block_size=16, raw=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.shape != (img_size, img_size):
        gray = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)

    h, w = gray.shape
    features = []

    if raw:
        gray = gray.astype(np.float32) / 255.0
        window = np.outer(np.hanning(block_size), np.hanning(block_size)).astype(np.float32)

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = gray[i:i+block_size, j:j+block_size]

            if block.shape != (block_size, block_size):
                continue

            if raw:
                block = (block - block.mean()) * window
                f = np.fft.fft2(block)
                fshift = np.fft.fftshift(f)
                magnitude = np.log1p(np.abs(fshift))
                features.extend(magnitude.flatten())
            else:
                f = np.fft.fft2(block)
                fshift = np.fft.fftshift(f)
                magnitude = np.abs(fshift)
                magnitude += 1e-8

                energy = np.sum(magnitude**2)
                prob = magnitude / np.sum(magnitude)
                entropy = -np.sum(prob * np.log(prob))
                mean_val = np.mean(magnitude)
                std_val = np.std(magnitude)

                center = block_size // 2
                low_freq = magnitude[center-4:center+4, center-4:center+4]
                low_energy = np.sum(low_freq**2)
                high_energy = energy - low_energy

                features.extend([energy, entropy, mean_val, std_val, low_energy, high_energy])

    return np.asarray(features, dtype=np.float32) if raw else np.array(features)

def extract_gradcam_blockwise_fft(image, cam, img_size=96, block_size=16, raw=False, min_attention=0.25, attention_strength=1.75):
    image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    cam = cv2.resize(cam.astype(np.float32), (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    
    features = []
    if raw:
        window = np.outer(np.hanning(block_size), np.hanning(block_size)).astype(np.float32)

    for i in range(0, img_size, block_size):
        for j in range(0, img_size, block_size):
            block = gray[i:i + block_size, j:j + block_size]
            cam_block = cam[i:i + block_size, j:j + block_size]

            if block.shape != (block_size, block_size):
                continue

            block_weight = min_attention + attention_strength * float(cam_block.mean())
            attention = min_attention + attention_strength * cam_block
            
            if raw:
                weighted_block = (block - block.mean()) * attention * window
                f = np.fft.fft2(weighted_block)
                fshift = np.fft.fftshift(f)
                magnitude = np.log1p(np.abs(fshift) * block_weight)
                features.extend(magnitude.flatten())
            else:
                weighted_block = block * attention
                f = np.fft.fft2(weighted_block)
                fshift = np.fft.fftshift(f)
                magnitude = np.abs(fshift) * block_weight
                magnitude += 1e-8

                energy = np.sum(magnitude ** 2)
                prob = magnitude / np.sum(magnitude)
                entropy = -np.sum(prob * np.log(prob))
                mean_val = np.mean(magnitude)
                std_val = np.std(magnitude)

                center = block_size // 2
                low_freq = magnitude[center - 4:center + 4, center - 4:center + 4]
                low_energy = np.sum(low_freq ** 2)
                high_energy = energy - low_energy

                features.extend([energy, entropy, mean_val, std_val, low_energy, high_energy])

    return np.asarray(features, dtype=np.float32)
