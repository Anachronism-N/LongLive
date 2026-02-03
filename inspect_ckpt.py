import torch
import sys

ckpt_path = "/commondocument/group2/LongLive/outputs/longlive_i2v_train/checkpoint_model_001600/model.pt"

print(f"Loading {ckpt_path}...")
try:
    state_dict = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state_dict, dict):
        print("Keys:", state_dict.keys())
        for k in state_dict.keys():
            if isinstance(state_dict[k], dict):
                 print(f"Key '{k}' is a dict with {len(state_dict[k])} items")
            elif hasattr(state_dict[k], 'shape'):
                 print(f"Key '{k}' is a tensor of shape {state_dict[k].shape}")
            else:
                 print(f"Key '{k}' is type {type(state_dict[k])}")
    else:
        print("State dict is not a dict, it is:", type(state_dict))
except Exception as e:
    print(f"Error loading: {e}")
