import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm

# --- YOLLAR ---
BASE_DIR = r"C:\Users\Eymen\Desktop\EKG\GÖREV 1"
BENCHMARK_DIR = os.path.join(BASE_DIR, "düzeltilmiş_resnet_v2", "benchmark")
OUTPUT_DIR = os.path.join(BASE_DIR, "ENSEMBLE")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['Normal', 'İletim', 'Ritim']

# Model Dosyaları
MODEL_PATHS = {
    "Conformer": os.path.join(BASE_DIR, "custom_transformer", "BEST_FINETUNED_AUGMENTED_CONFORMER_COLAB.pth"),
    "Res18_Hybrid": os.path.join(BASE_DIR, "düzeltilmiş_resnet_v2", "models_hybrid_final", "best_model.pth"),
    "Res18_Std": os.path.join(BASE_DIR, "düzeltilmiş_resnet_v2", "models_resnet18_std", "best_model.pth"),
    "Res101_Hybrid": os.path.join(BASE_DIR, "düzeltilmiş_resnet_v2", "models", "best_model.pth")
}

# --- MODELLERİ YÜKLE ---
# Not: Mimari sınıflarının (Conformer, ResNet101_1D vb.) kodda tanımlı olması gerekir.
# Eğer import hatası alırsan o modellerin class tanımlarını buraya yapıştırmalıyız.

def get_ensemble_predictions():
    print("🚀 Dört Dev Sahaya İniyor...")
    
    # Placeholder: Burada tüm modelleri load_state_dict ile yüklüyoruz
    # Önemli: Modellerin her birine 'outputs' üzerinden softmax uygulayacağız.
    
    # Test verisini hazırla
    files = glob.glob(os.path.join(BENCHMARK_DIR, "val_part*.pt"))
    test_ds = ConcatDataset([torch.load(f)['X'] for f in files]) # Şematik
    # ... (Burada DataLoader ile döngü başlıyor)

    print("📊 Modeller arası oylama yapılıyor...")
    # Formül: (Conf_Prob * 0.4) + (Res18_Prob * 0.3) + (Res101_Prob * 0.3)
    # Conformer'a daha yüksek ağırlık veriyoruz çünkü Recall'ı çok yüksek.