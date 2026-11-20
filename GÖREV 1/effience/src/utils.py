
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import torch
from . import config

def plot_training_history(train_losses, train_accs, val_losses, val_accs, output_dir):
    plt.figure(figsize=(12, 5))
    
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_plot.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=config.CLASS_NAMES, 
                yticklabels=config.CLASS_NAMES)
    plt.title('Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

from sklearn.metrics import classification_report

def plot_ecg_sample(signal, true_label, pred_label, index, output_dir):
    """
    Plots Lead I, II, III for a single sample.
    signal: (12, L) numpy array
    true_label: int
    pred_label: int
    """
    leads_to_plot = [0, 1, 2] # I, II, III
    lead_names = ["Lead I", "Lead II", "Lead III"]
    
    true_name = config.CLASS_NAMES[true_label]
    pred_name = config.CLASS_NAMES[pred_label]
    
    # Color: Green if correct, Red if wrong
    is_correct = (true_label == pred_label)
    color = 'green' if is_correct else 'red'
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    # Title matching the user's requested format
    # "Model: ... \n True: ... | Pred: ..."
    fig.suptitle(f"Model: EfficientNet1D\nTrue: ['{true_name}'] | Pred: ['{pred_name}']", 
                 fontsize=14, color=color, weight='bold')
    
    # Plot Leads
    for i, ax in enumerate(axes):
        lead_idx = leads_to_plot[i]
        
        # Plot center 2000 samples for better visibility (or full if short)
        plot_len = 2000
        if signal.shape[1] > plot_len:
            start = (signal.shape[1] - plot_len) // 2
            segment = signal[lead_idx, start:start+plot_len]
        else:
            segment = signal[lead_idx, :]
            
        ax.plot(segment, linewidth=1.2)
        ax.set_title(lead_names[i], fontsize=10, weight='bold')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_ylabel("mV")
        
        # Hide x ticks for top plots
        if i < 2:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel("Time (samples)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save
    prefix = "CORRECT" if is_correct else "WRONG"
    save_path = os.path.join(output_dir, f"{prefix}_{index}_{true_name}_vs_{pred_name}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
