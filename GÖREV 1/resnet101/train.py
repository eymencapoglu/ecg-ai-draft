
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import numpy as np
import wfdb
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
import sys

# Add parent path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from effience.src.config import *
from resnet101.model import ResNet101_1D
from resnet101.utils import clean_signal

# Dataset Class for Training (On-the-fly loading & cleaning)
class ResNetTrainDataset(Dataset):
    def __init__(self, manifest_path, transform=True):
        self.df = pd.read_csv(manifest_path)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['file_path']
        label = int(row['label'])
        
        try:
            # Load basic
            data, _ = wfdb.rdsamp(path)
            data = data.T.copy() 
            
            # Normalize
            if np.std(data) != 0:
                data = (data - np.mean(data)) / (np.std(data) + 1e-6)
            else:
                data = np.zeros_like(data)
                
            # Pad/Crop
            if data.shape[1] > input_length:
                start = (data.shape[1] - input_length) // 2
                data = data[:, start:start+input_length]
            else:
                pad_len = input_length - data.shape[1]
                data = np.pad(data, ((0,0), (0, pad_len)))
                
            # CLEANING (Bandpass)
            data = clean_signal(data, fs=SAMPLE_RATE)
            
            # Augmentation (Optional, if transform=True)
            # We can reuse effience augmentation logic or strict it.
            # For now, let's keep it simple or copy paste simple aug.
            if self.transform:
               if np.random.rand() < 0.5: # Noise
                   data = data + np.random.normal(0, 0.01, data.shape)
               if np.random.rand() < 0.5: # Scale
                   data = data * np.random.uniform(0.9, 1.1)
                   
            return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
            
        except:
             return torch.zeros((12, input_length), dtype=torch.float32), torch.tensor(0, dtype=torch.long)

# Dataset Class for Benchmark (Pre-processed .pt files)
class BenchmarkDataset(Dataset):
    def __init__(self, benchmark_dir):
        self.benchmark_dir = benchmark_dir
        self.manifest_path = os.path.join(benchmark_dir, 'manifest.csv')
        self.df = pd.read_csv(self.manifest_path)
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['filename']
        path = os.path.join(self.benchmark_dir, filename)
        
        item = torch.load(path)
        return item['signal'], item['label']

def train():
    # Config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training ResNet101 on {device}")
    
    batch_size = 32
    num_epochs = 10 # Increased slightly for ResNet
    lr = 1e-3
    
    # Paths
    train_manifest = os.path.join(BASE_DIR, 'resnet101', 'train_manifest.csv')
    benchmark_dir = os.path.join(BASE_DIR, 'benchmark')
    output_dir = os.path.join(BASE_DIR, 'resnet101', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Data Loaders
    print("Loading Data...")
    if not os.path.exists(train_manifest):
        print("Train manifest not found. Run prepare_data.py first.")
        return
        
    train_ds = ResNetTrainDataset(train_manifest, transform=True)
    benchmark_ds = BenchmarkDataset(benchmark_dir)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(benchmark_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Train Size: {len(train_ds)}")
    print(f"Test (Benchmark) Size: {len(benchmark_ds)}")
    
    # Model
    model = ResNet101_1D(num_classes=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for signals, labels in loop:
            signals, labels = signals.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * signals.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loop.set_postfix(loss=loss.item())
            
        train_loss = running_loss / len(train_ds)
        train_acc = correct / total
        
        # Validation on Benchmark
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for signals, labels in val_loader:
                signals, labels = signals.to(device), labels.to(device)
                outputs = model(signals)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * signals.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_loss = val_running_loss / len(benchmark_ds)
        val_acc = val_correct / val_total
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, 'resnet101_best.pth'))
            print("Saved Best Model")
            
    print("Training Complete.")

if __name__ == "__main__":
    train()
