import xgboost as xgb
import torch

def build_xgboost_model(num_classes, xgb_params, device="cpu"):
    params = xgb_params.copy()
    params["num_class"] = num_classes
    
    if device == "cuda" or (device == "auto" and torch.cuda.is_available()):
        params["device"] = "cuda"
        
    return xgb.XGBClassifier(**params)
