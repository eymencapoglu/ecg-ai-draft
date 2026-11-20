
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import wfdb
import ast
from . import config

def get_labels_from_header(header_path):
    try:
        # Example line in .hea: "# Dx: 270492004,164889003"
        with open(header_path, 'r') as f:
            for line in f:
                if line.startswith('# Dx:') or line.startswith('#Dx:'):
                    codes = line.split(':')[-1].strip().split(',')
                    return [c.strip() for c in codes]
    except:
        return []
    return []

def apply_augmentations(data):
    """
    Apply random augmentations to 1D ECG signal.
    data: (12, L) numpy array
    """
    # 1. Random Gaussian Noise
    if np.random.rand() < 0.5:
        sigma = 0.01 * np.random.rand()
        noise = np.random.normal(0, sigma, data.shape)
        data = data + noise
        
    # 2. Random Scaling (Amplitude)
    if np.random.rand() < 0.5:
        scale_factor = np.random.uniform(0.9, 1.1)
        data = data * scale_factor
        
    # 3. Random Shift (Time)
    if np.random.rand() < 0.5:
        shift = np.random.randint(-100, 100)
        if shift > 0:
            data = np.pad(data[:, :-shift], ((0,0), (shift, 0)), mode='constant')
        elif shift < 0:
            data = np.pad(data[:, -shift:], ((0,0), (0, -shift)), mode='constant')
            
    return data

class MultiSourceECGDataset(Dataset):
    def __init__(self, use_ds1=True, use_ds2=True, use_ds3=True, transform=False):
        self.samples = [] 
        self.transform = transform
        
        # Load Dataset 1
        if use_ds1:
            print("Loading Dataset 1 index...")
            rec_path = os.path.join(config.DATASET1_DIR, "WFDBRecords")
            if os.path.exists(rec_path):
                cnt = 0
                for root, dirs, files in os.walk(rec_path):
                    for file in files:
                        if file.endswith(".hea"):
                            full_path = os.path.join(root, file)
                            codes = get_labels_from_header(full_path)
                            lbl = self.map_codes(codes, config.SNOMED_MAPPING) 
                            if lbl is not None:
                                base_path = full_path[:-4]
                                self.samples.append((base_path, lbl, 'ds1'))
                                cnt += 1
                print(f"Dataset 1: Found {cnt} valid samples.")

        # Load Dataset 2 (Generic Scan)
        if use_ds2:
            print(f"Checking Dataset 2 at {os.path.join(config.BASE_DIR, 'dataset 2')}...")
            ds2_path = os.path.join(config.BASE_DIR, "dataset 2")
            if os.path.exists(ds2_path):
                cnt = 0
                for root, dirs, files in os.walk(ds2_path):
                    for file in files:
                        if file.endswith(".hea"):
                            full_path = os.path.join(root, file)
                            codes = get_labels_from_header(full_path)
                            # Assuming DS2 uses same SNOMED codes or similar
                            # If completely different, this mapping needs update.
                            # Using SNOMED mapping as fallback.
                            lbl = self.map_codes(codes, config.SNOMED_MAPPING) 
                            if lbl is not None:
                                base_path = full_path[:-4]
                                self.samples.append((base_path, lbl, 'ds2'))
                                cnt += 1
                print(f"Dataset 2: Found {cnt} valid samples.")
            else:
                print("Dataset 2 path not found.")

        # Load Dataset 3 (PTB-XL)
        if use_ds3:
            print("Loading Dataset 3 index...")
            try:
                df = pd.read_csv(os.path.join(config.DATASET3_DIR, 'ptbxl_database.csv'), index_col='ecg_id')
                df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))
                
                cnt = 0
                for idx, row in df.iterrows():
                    codes = list(row.scp_codes.keys())
                    lbl = self.map_codes(codes, config.SCP_MAPPING) 
                    if lbl is not None:
                        path = os.path.join(config.DATASET3_DIR, row['filename_hr']) 
                        self.samples.append((path, lbl, 'ds3'))
                        cnt += 1
                print(f"Dataset 3: Found {cnt} valid samples.")
            except Exception as e:
                print(f"Warning: Could not load Dataset 3: {e}")
            
    def map_codes(self, codes, mapping):
        # Specific logic based on the strict hierarchy: 
        # Arrhythmia (1) > Block (2) > Normal (0)
        
        detected_classes = set()
        for c in codes:
            if str(c) in mapping:
                detected_classes.add(mapping[str(c)])
        
        if 1 in detected_classes: # Arrhythmia
            return 1
        if 2 in detected_classes: # Block
            return 2
        if 0 in detected_classes: # Normal
            return 0
            
        return None 
            
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, source = self.samples[idx]
        
        try:
            data, _ = wfdb.rdsamp(path)
            data = data.T.copy() 
            
            # Normalize (Z-score standard) with epsilon
            if np.std(data) != 0:
                data = (data - np.mean(data)) / (np.std(data) + 1e-6)
            else:
                data = np.zeros_like(data)
                
            # Pad/Crop
            if data.shape[1] > config.input_length:
                start = (data.shape[1] - config.input_length) // 2
                data = data[:, start:start+config.input_length]
            else:
                pad_len = config.input_length - data.shape[1]
                data = np.pad(data, ((0,0), (0, pad_len)))
            
            # Augmentation (only if transform is enabled)
            if self.transform:
                data = apply_augmentations(data)
                
            return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
            
        except Exception as e:
            return torch.zeros((12, config.input_length), dtype=torch.float32), torch.tensor(0, dtype=torch.long)
