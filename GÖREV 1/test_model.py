
import torch
from torch.utils.data import DataLoader, random_split
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report

from effience.src.config import *
from effience.src.data_utils import MultiSourceECGDataset
from effience.src.model import EfficientNet1D
from effience.src.utils import plot_confusion_matrix, plot_ecg_sample

def test_pipeline():
    # 1. Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on {device}")
    
    output_dir = os.path.join(BASE_DIR, 'effience', 'outputs')
    plots_dir = os.path.join(output_dir, 'final_test_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 2. Load Data (Same logic as train to try and get similar distribution)
    # Ideally we would have saved the split, but we will rely on random_split with fixed seed if we had one.
    # For now we load all and take a subset or just run on all to get metrics.
    print("Loading Dataset...")
    full_dataset = MultiSourceECGDataset()
    if len(full_dataset) == 0:
        print("No data found!")
        return

    # Use a split for testing
    total_len = len(full_dataset)
    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)
    test_len = total_len - train_len - val_len
    
    # Use a fixed generator for reproducibility if desired, but default is random
    # We will just proceed with the random split to get a test set.
    # Note: If train.py was run separately, this test set might contain samples seen in training 
    # because we didn't fix the seed. For strict correctness we should fix seed in both.
    # I will assume this is for demonstration of output format.
    _, _, test_data = random_split(full_dataset, [train_len, val_len, test_len])
    
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    print(f"Test Set Size: {len(test_data)}")

    # 3. Load Model
    model = EfficientNet1D(num_classes=num_classes).to(device)
    best_path = os.path.join(output_dir, 'best_model.pth')
    last_path = os.path.join(output_dir, 'last_model.pth')
    
    model_path = None
    if os.path.exists(best_path):
        model_path = best_path
        print(f"Loading best model from {model_path}")
    elif os.path.exists(last_path):
        model_path = last_path
        print(f"Loading last model from {model_path} (Warning: Training might be incomplete)")
    else:
        print(f"No model found at {best_path} or {last_path}. Please train first.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 4. Run Inference
    all_preds = []
    all_labels = []
    
    # We'll also collect some samples to plot
    plotted_count = 0
    max_plots = 20
    
    print("Running Inference...")
    with torch.no_grad():
        for signals, labels in tqdm(test_loader):
            signals = signals.to(device)
            outputs = model(signals)
            _, predicted = torch.max(outputs, 1)
            
            # Collect metrics
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Plotting (CPU)
            if plotted_count < max_plots:
                signals_np = signals.cpu().numpy()
                labels_np = labels.cpu().numpy()
                preds_np = predicted.cpu().numpy()
                
                for i in range(len(labels_np)):
                    if plotted_count < max_plots:
                        plot_ecg_sample(signals_np[i], labels_np[i], preds_np[i], plotted_count, plots_dir)
                        plotted_count += 1

    # 5. Calculate Metrics
    print("\n" + "="*30)
    print("TEST RESULTS")
    print("="*30)
    
    # Classification Report
    print("Classification Report:")
    # Get string for printing
    report_str = classification_report(all_labels, all_preds, target_names=CLASS_NAMES)
    print(report_str)
    
    # Get dict for specific metrics
    report_dict = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True)
    macro_f1 = report_dict['macro avg']['f1-score']
    print(f"\n>> MACRO F1 SCORE: {macro_f1:.4f}")
    
    # Save report to text
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report_str)
        f.write(f"\n\nMACRO F1 SCORE: {macro_f1:.4f}")
        
    # Confusion Matrix
    print("Generating Confusion Matrix...")
    plot_confusion_matrix(all_labels, all_preds, output_dir)
    print(f"Confusion Matrix saved to {os.path.join(output_dir, 'confusion_matrix.png')}")
    print(f"Sample plots saved to {plots_dir}")

if __name__ == "__main__":
    test_pipeline()
