import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
from ..data.transforms import preprocess_for_resnet

class GradCAM:
    def __init__(self, model, target_layer_name, img_size, device, imagenet_mean, imagenet_std):
        self.model = model
        self.img_size = img_size
        self.device = device
        self.imagenet_mean = imagenet_mean
        self.imagenet_std = imagenet_std
        
        self.gradients = None
        self.activations = None
        
        target_layer = dict([*self.model.named_modules()])[target_layer_name]
        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, inputs, output):
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def get_cam(self, image, label=None, target_mode="predicted"):
        image_tensor = preprocess_for_resnet(image, self.img_size, self.device, self.imagenet_mean, self.imagenet_std)
        self.gradients = None
        self.activations = None

        self.model.zero_grad(set_to_none=True)
        output = self.model(image_tensor)

        if target_mode == "ground_truth" and label is not None:
            target_class = int(label)
        else:
            target_class = int(output.argmax(dim=1).item())

        score = output[:, target_class].sum()
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()

        cam_min = cam.min()
        cam_max = cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam.astype(np.float32)

def load_or_compute_cam(image, label, img_path, split_name, class_name, cam_cache_dir, gradcam_instance, target_mode="predicted"):
    stem = Path(img_path).stem
    save_path = Path(cam_cache_dir) / split_name / class_name / f"{stem}.npy"

    if save_path.exists():
        return np.load(save_path)

    cam = gradcam_instance.get_cam(image, label, target_mode)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, cam)
    return cam
