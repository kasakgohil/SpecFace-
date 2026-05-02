import os
import yaml

class Config:
    def __init__(self, config_path="configs/default_config.yaml"):
        with open(config_path, "r") as f:
            self.yaml_config = yaml.safe_load(f)
            
        # Paths
        self.DATA_DIR = self.yaml_config.get("data_dir", r"C:\work\Capstone\stl\stl10_binary")
        self.DATASET_DIR = self.yaml_config.get("dataset_dir", r"C:\work\Capstone\data\preprocessed")
        self.ARTIFACT_DIR = self.yaml_config.get("artifact_dir", "artifacts")

        # Directories
        self.PREPROCESSOR_DIR = os.path.join(self.ARTIFACT_DIR, "preprocessors")
        self.LOG_DIR = os.path.join(self.ARTIFACT_DIR, "logs")
        self.PYTORCH_MODEL_DIR = os.path.join(self.ARTIFACT_DIR, "models", "pytorch")
        self.XGBOOST_MODEL_DIR = os.path.join(self.ARTIFACT_DIR, "models", "xgboost")
        
        for d in [self.PREPROCESSOR_DIR, self.LOG_DIR, self.PYTORCH_MODEL_DIR, self.XGBOOST_MODEL_DIR]:
            os.makedirs(d, exist_ok=True)

        # Dataset params
        self.IMG_SIZE = self.yaml_config.get("img_size", 96)
        self.BLOCK_SIZE = self.yaml_config.get("block_size", 16)
        self.TRAIN_RATIO = self.yaml_config.get("train_ratio", 0.7)
        self.VAL_RATIO = self.yaml_config.get("val_ratio", 0.15)
        self.GAUSSIAN_KSIZE = self.yaml_config.get("gaussian_ksize", 3)

        # Training params
        self.EPOCHS = self.yaml_config.get("epochs", 80)
        self.LR = self.yaml_config.get("learning_rate", 3e-4)
        self.RESNET_LR = self.yaml_config.get("resnet_lr", 1e-4)
        self.FFT_GLOBAL_LR = self.yaml_config.get("fft_global_lr", 1e-3)
        self.FFT_GLOBAL_WD = self.yaml_config.get("fft_global_weight_decay", 1e-5)
        self.FFT_BLOCKWISE_WD = self.yaml_config.get("fft_blockwise_weight_decay", 1e-4)
        self.LABEL_SMOOTHING = self.yaml_config.get("label_smoothing", 0.05)
        self.BATCH_SIZE = self.yaml_config.get("batch_size", 512)
        self.NUM_WORKERS = self.yaml_config.get("num_workers", 8)
        self.PATIENCE = self.yaml_config.get("patience", 15)
        self.EARLY_STOPPING_ROUNDS = self.yaml_config.get("early_stopping_rounds", 75)

        # PCA components
        self.PCA_COMPONENTS = self.yaml_config.get("pca_components", 512)
        
        # Grad-CAM specific
        self.CAM_TARGET_MODE = self.yaml_config.get("cam_target_mode", "predicted")
        self.MIN_ATTENTION = self.yaml_config.get("min_attention", 0.25)
        self.ATTENTION_STRENGTH = self.yaml_config.get("attention_strength", 1.75)
        
        # XGBoost params
        self.XGB_PARAMS = self.yaml_config.get("xgb", {})
        
        # Seed
        self.SEED = self.yaml_config.get("seed", 42)

config = Config()
