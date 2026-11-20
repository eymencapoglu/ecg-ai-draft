
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import random_split
import sys

# Add parent dir to path to access effience.src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from effience.src.config import *
from effience.src.data_utils import MultiSourceECGDataset
from resnet101.utils import clean_signal

def prepare_data():
    print("Initializing Dataset...")
    # Load all data without transforms (we will clean them)
    full_dataset = MultiSourceECGDataset(use_ds1=True, use_ds2=True, use_ds3=True, transform=False)
    
    total_len = len(full_dataset)
    print(f"Total samples found: {total_len}")
    
    if total_len < 5000:
        print("Error: Not enough data for 5k test set.")
        return

    # Split: 5000 for Test (Benchmark), rest for Train
    test_len = 5000
    train_len = total_len - test_len
    
    # Random split
    generator = torch.Generator().manual_seed(42)
    train_subset, test_subset = random_split(full_dataset, [train_len, test_len], generator=generator)
    
    # Save Benchmark Data
    benchmark_dir = os.path.join(BASE_DIR, 'benchmark')
    os.makedirs(benchmark_dir, exist_ok=True)
    
    print(f"Processing Benchmark Set ({test_len} samples)...")
    
    benchmark_records = []
    
    # We will save each cleaned tensor or a large file if it fits. 
    # 5000 * 12 * 5000 * 4 bytes ~= 1.2 GB. It might fit in memory but safer to save individually or chunks.
    # User said "benchmark diye bir klasör oluşturup oraya kaydet her seferinde test için o verileri kullanalım"
    # Saving individual files is cleaner for loading on demand, or a few chunks.
    # Let's save individual .pt files.
    
    for i, idx in enumerate(tqdm(test_subset.indices)):
        path, label, source = full_dataset.samples[idx]
        
        # Load and clean
        # access via dataset methods to handle loading/padding logic consistently
        # But wait, dataset.__getitem__ returns tensor and applies logic.
        # But we need to apply CLEANING which is NOT in the original dataset.
        
        # So we manually load using the dataset logic but ADD cleaning.
        # Ideally we refactor data_utils but avoiding big changes on `effience` as per instruction scope.
        # We will use dataset.__getitem__ then apply cleaning on the tensor.
        # Note: clean_signal expects numpy (channels, length).
        
        # Get raw data (tensor)
        raw_tensor, _ = full_dataset[idx] 
        # Force copy to ensure positive strides from the start
        raw_np = raw_tensor.numpy().copy()
        
        # Apply Clean
        cleaned_np = clean_signal(raw_np, fs=SAMPLE_RATE)
        # Force copy again just in case filtfilt did something weird
        cleaned_tensor = torch.tensor(cleaned_np.copy(), dtype=torch.float32)
        
        # Save
        filename = f"test_{i:04d}.pt"
        save_path = os.path.join(benchmark_dir, filename)
        torch.save({'signal': cleaned_tensor, 'label': label}, save_path)
        
        benchmark_records.append({
            'filename': filename,
            'original_path': path,
            'label': label,
            'class_name': CLASS_NAMES[int(label)],
            'source': source
        })
        
    # Save Manifest
    df = pd.DataFrame(benchmark_records)
    df.to_csv(os.path.join(benchmark_dir, 'manifest.csv'), index=False)
    print(f"Benchmark set saved to {benchmark_dir}")
    
    # Training Data
    # For training data, we don't need to save all tensors (would be huge, 45k * 1.2/5 ~= 10GB).
    # We will just save the INDICES/Manifest for training, and apply cleaning On-The-Fly during training.
    
    print(f"Saving Train Manifest ({train_len} samples)...")
    train_records = []
    
    for idx in train_subset.indices:
        path, label, source = full_dataset.samples[idx]
        train_records.append({
            'file_path': path,
            'label': label,
            'source': source
        })
        
    train_df = pd.DataFrame(train_records)
    train_manifest_path = os.path.join(BASE_DIR, 'resnet101', 'train_manifest.csv')
    train_df.to_csv(train_manifest_path, index=False)
    print(f"Train manifest saved to {train_manifest_path}")

if __name__ == "__main__":
    prepare_data()
