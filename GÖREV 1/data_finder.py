
import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import wfdb
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

from effience.src.config import *
from effience.src.model import EfficientNet1D
from effience.src.utils import plot_confusion_matrix

class ManifestDataset(Dataset):
    def __init__(self, manifest_path):
        self.df = pd.read_csv(manifest_path)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['file_path']
        label = int(row['label'])
        
        try:
            data, _ = wfdb.rdsamp(path)
            data = data.T 
            
            if np.std(data) != 0:
                data = (data - np.mean(data)) / (np.std(data) + 1e-6)
            else:
                data = np.zeros_like(data)
                
            if data.shape[1] > input_length:
                start = (data.shape[1] - input_length) // 2
                data = data[:, start:start+input_length]
            else:
                pad_len = input_length - data.shape[1]
                data = np.pad(data, ((0,0), (0, pad_len)))
                
            return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
        except:
             return torch.zeros((12, input_length), dtype=torch.float32), torch.tensor(0, dtype=torch.long)

def benchmark_model(model_name="MyModel"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking on {device}")
    
    output_dir = os.path.join(BASE_DIR, 'effience', 'outputs')
    manifest_path = os.path.join(output_dir, 'validation_manifest.csv')
    
    if not os.path.exists(manifest_path):
        print("Manifest not found! Run training first.")
        return

    # Load fixed validation set
    val_dataset = ManifestDataset(manifest_path)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"Loaded {len(val_dataset)} samples from manifest.")

    # Load Model
    model = EfficientNet1D(num_classes=num_classes).to(device)
    model_path = os.path.join(output_dir, 'best_model.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        print("Model checkpoint not found.")
        return
        
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for signals, labels in val_loader:
            signals = signals.to(device)
            outputs = model(signals)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    print(f"\nModel: {model_name}")
    print(f"Validation Accuracy: {acc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

if __name__ == "__main__":
    benchmark_model()
