
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from sklearn.metrics import f1_score

from effience.src.config import *
from effience.src.data_utils import MultiSourceECGDataset
from effience.src.model import EfficientNet1D
from effience.src.utils import plot_training_history, plot_confusion_matrix

warnings.filterwarnings("ignore")

def train_model():
    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Dataset
    print("Initializing Data Loader...")
    # We will instantiate separate datasets for train and val to control transforms
    # But splitting must be consistent.
    # Approach: Create one full dataset without transforms to get indices, then use those indices on Subsets.
    # Actually, custom Subset wrapper is better to handle transform override.
    
    full_dataset_no_aug = MultiSourceECGDataset(transform=False)
    
    # Split Train/Val/Test (80-10-10)
    total_len = len(full_dataset_no_aug)
    if total_len == 0:
        print("Error: No data found. Check paths and mappings.")
        return

    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)
    test_len = total_len - train_len - val_len
    
    # Random Split
    generator = torch.Generator().manual_seed(42) # Reproducibility
    train_subset, val_subset, test_subset = random_split(full_dataset_no_aug, [train_len, val_len, test_len], generator=generator)
    
    # Save Validation Set Manifest (CSV)
    # This ensures we have a fixed list of files for benchmarking other models.
    output_dir = os.path.join(BASE_DIR, 'effience', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    print("Saving Validation Manifest...")
    val_indices = val_subset.indices
    val_records = []
    
    for idx in val_indices:
        # Access original sample info directly to avoid loading signal
        path, label, source = full_dataset_no_aug.samples[idx]
        val_records.append({
            'file_path': path, 
            'label': label,
            'source': source,
            'class_name': CLASS_NAMES[label]
        })
        
    val_df = pd.DataFrame(val_records)
    val_manifest_path = os.path.join(output_dir, 'validation_manifest.csv')
    val_df.to_csv(val_manifest_path, index=False)
    print(f"Validation Manifest saved to {val_manifest_path} ({len(val_df)} samples)")

    # Create Wrapper for Augmentation on Train
    class AugmentedSubset(torch.utils.data.Dataset):
        def __init__(self, subset, transform=False):
            self.subset = subset
            self.transform = transform
            self.dataset_Ref = subset.dataset
            
        def __getitem__(self, idx):
            # Access underlying dataset item, apply transform is tricky because logic is in dataset.__getitem__
            # Current MultiSourceECGDataset.__getitem__ uses self.transform.
            # We need to temporarily force self.transform of the parent? No, that's not thread safe.
            # Better: Modify full_dataset or pass dataset with transform=True for training.
            
            # Simple Hack: We know the underlying dataset class has logic.
            # We can create two datasets: one with transform=True, one False.
            # And use the SAME split indices.
            pass
            
    # BETTER APPROACH:
    # 1. Get indices from random_split.
    # 2. Create TrainDataset(transform=True) and ValDataset(transform=False).
    # 3. Use Subset(TrainDataset, train_indices) etc.
    
    train_indices = train_subset.indices
    val_indices = val_subset.indices
    test_indices = test_subset.indices
    
    train_dataset = MultiSourceECGDataset(transform=True) # Augmentation ON
    val_dataset   = MultiSourceECGDataset(transform=False) # Augmentation OFF
    
    train_data = torch.utils.data.Subset(train_dataset, train_indices)
    val_data   = torch.utils.data.Subset(val_dataset, val_indices)
    test_data  = torch.utils.data.Subset(val_dataset, test_indices) # Use val_dataset (no aug) for test
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    print(f"Data Loaded. Train: {train_len}, Val: {val_len}, Test: {test_len}")

    # Model
    model = EfficientNet1D(num_classes=num_classes).to(device)
    
    # Regularization: Label Smoothing & Weight Decay
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # Helps with overfitting
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2) # Added regularzation
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # Metrics
    best_val_loss = float('inf')
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    output_dir = os.path.join(BASE_DIR, 'effience', 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    # Training Loop
    print(f"Starting Training for {num_epochs} epochs...")
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
            
            # Check for NaN loss
            if torch.isnan(loss):
                print("Warning: NaN loss detected!")
                continue
                
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item() * signals.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loop.set_postfix(loss=loss.item())
            
        epoch_loss = running_loss / train_len if train_len > 0 else 0
        epoch_acc = correct / total if total > 0 else 0
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # Validation
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        val_all_preds = []
        val_all_labels = []
        
        with torch.no_grad():
            for signals, labels in val_loader:
                signals, labels = signals.to(device), labels.to(device)
                outputs = model(signals)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * signals.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_all_preds.extend(predicted.cpu().numpy())
                val_all_labels.extend(labels.cpu().numpy())
        
        val_loss = val_running_loss / val_len if val_len > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0
        val_f1 = f1_score(val_all_labels, val_all_preds, average='macro')
        
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1 (Macro): {val_f1:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            # Save best model
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
            
        # Also save the latest model every epoch (overwrite)
        torch.save(model.state_dict(), os.path.join(output_dir, 'last_model.pth'))
            
    # Visualize Training
    plot_training_history(train_losses, train_accs, val_losses, val_accs, output_dir)
    print("Training Complete. Testing and Generating Plots...")
    
    # Testing
    # Load best if exists, else load last
    best_path = os.path.join(output_dir, 'best_model.pth')
    last_path = os.path.join(output_dir, 'last_model.pth')
    
    if os.path.exists(best_path):
        print(f"Loading best model from {best_path}")
        model.load_state_dict(torch.load(best_path))
    elif os.path.exists(last_path):
        print(f"Warning: Best model not found (maybe NaN loss?). Loading last model from {last_path}")
        model.load_state_dict(torch.load(last_path))
    else:
        print("Error: No model checkpoint found!")
        return
    
    all_preds = []
    all_labels = []
    
    # Ensure plots directory exists
    plots_dir = os.path.join(output_dir, 'sample_predictions')
    os.makedirs(plots_dir, exist_ok=True)
    
    sample_count = 0
    max_samples_to_plot = 50 # Plot 50 samples to avoid flooding
    
    with torch.no_grad():
        for signals, labels in tqdm(test_loader, desc="Testing"):
            signals = signals.to(device)
            outputs = model(signals)
            _, predicted = torch.max(outputs, 1)
            
            # Save stats
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Plot samples (cpu for plotting)
            signals_np = signals.cpu().numpy()
            labels_np = labels.cpu().numpy()
            preds_np = predicted.cpu().numpy()
            
            for i in range(len(labels_np)):
                if sample_count < max_samples_to_plot:
                    from effience.src.utils import plot_ecg_sample
                    plot_ecg_sample(signals_np[i], labels_np[i], preds_np[i], sample_count, plots_dir)
                    sample_count += 1
            
    # Confusion Matrix
    plot_confusion_matrix(all_labels, all_preds, output_dir)
    print(f"Confusion Matrix saved to {output_dir}")
    print(f"Sample plots saved to {plots_dir}")

if __name__ == '__main__':
    train_model()
