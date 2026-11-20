
import torch
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import sys

# Add parent path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from effience.src.config import *
from resnet101.model import ResNet101_1D
from resnet101.train import BenchmarkDataset

def plot_confusion_matrix_custom(y_true, y_pred, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, 
                yticklabels=CLASS_NAMES)
    plt.title('ResNet101 Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def test_resnet():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing ResNet101 on {device}")
    
    # Paths
    benchmark_dir = os.path.join(BASE_DIR, 'benchmark')
    model_dir = os.path.join(BASE_DIR, 'resnet101', 'outputs')
    plots_dir = os.path.join(model_dir, 'test_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    best_model_path = os.path.join(model_dir, 'resnet101_best.pth')
    
    if not os.path.exists(best_model_path):
        print(f"Error: Model not found at {best_model_path}")
        return

    # Load Data
    print("Loading Benchmark Data...")
    test_ds = BenchmarkDataset(benchmark_dir)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)
    print(f"Test Samples: {len(test_ds)}")
    
    # Load Model
    model = ResNet101_1D(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("Running Inference...")
    with torch.no_grad():
        for signals, labels in tqdm(test_loader):
            signals = signals.to(device)
            outputs = model(signals)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Metrics
    print("\n" + "="*30)
    print("TEST RESULTS (ResNet101)")
    print("="*30)
    
    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES)
    print(report)
    
    with open(os.path.join(model_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
        
    print("Generating Confusion Matrix...")
    plot_confusion_matrix_custom(all_labels, all_preds, model_dir)
    print(f"Results saved to {model_dir}")

if __name__ == "__main__":
    test_resnet()
