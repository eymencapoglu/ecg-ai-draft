import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset, Dataset
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
import glob
from tqdm import tqdm
import warnings
import math
import optuna

warnings.filterwarnings("ignore")

# =====================================================================
# --- AYARLAR (LOCAL VEYA COLAB SEÇİMİ) ---
# =====================================================================

# 1. SEÇENEK: YEREL (LOCAL - RTX 5060 Ti) İÇİN YOLLAR
DATA_SOURCE_DIR = r"C:\Users\Eymen\Desktop\EKG\GÖREV 1\düzeltilmiş_resnet_v2"
TRAIN_DIR = os.path.join(DATA_SOURCE_DIR, "processed_train")
BENCHMARK_DIR = os.path.join(DATA_SOURCE_DIR, "benchmark")
OUTPUT_DIR = r"C:\Users\Eymen\Desktop\EKG\GÖREV 1\custom_transformer"
PRETRAINED_BRAIN_PATH = os.path.join(OUTPUT_DIR, "pretrained_ekg_brain.pth") 

# 2. SEÇENEK: COLAB İÇİN YOLLAR (Eğer Colab'da isen üsttekileri yoruma al, alttakileri aç)
# DATA_SOURCE_DIR = '/content/ekg_data/ekg data'
# TRAIN_DIR = os.path.join(DATA_SOURCE_DIR, "processed_train")
# BENCHMARK_DIR = os.path.join(DATA_SOURCE_DIR, "benchmark")
# OUTPUT_DIR = '/content/drive/MyDrive/ekg_finetune_results' 
# PRETRAINED_BRAIN_PATH = '/content/drive/MyDrive/pretrained_ekg_brain.pth' 

os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64 # Local için 64 ideal. Colab'da isen 128 yapabilirsin.
NUM_CLASSES = 3

# =====================================================================
# --- YENİ: DİNAMİK VERİ ARTIRMA (ON-THE-FLY AUGMENTATION) ---
# =====================================================================
class AugmentedECGDataset(Dataset):
    def __init__(self, base_dataset, transform=False):
        self.base = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        x = x.clone() # Orijinal RAM'deki veriyi bozmamak için kopyalıyoruz
        
        # ResNet101'in sırrı burası: Her okumada %50 ihtimalle klonla ve değiştir!
        if self.transform:
            if torch.rand(1).item() < 0.5: # Noise (Gürültü Ekle)
                noise = torch.randn_like(x) * 0.01
                x = x + noise
            if torch.rand(1).item() < 0.5: # Scale (Ölçeklendir/Boyunu Uzat-Kısalt)
                scale = torch.empty(1).uniform_(0.9, 1.1).item()
                x = x * scale
                
        return x, y

# =====================================================================

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.weight = weight 
        self.gamma = gamma   

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

class Conformer_Classifier(nn.Module):
    def __init__(self, in_chans=12, embed_dim=128, depth=4, heads=4, dropout=0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_chans, embed_dim // 2, kernel_size=15, stride=5, padding=7),
            nn.BatchNorm1d(embed_dim // 2), nn.GELU(),
            nn.Conv1d(embed_dim // 2, embed_dim, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(embed_dim), nn.GELU()
        )
        self.num_patches = 250
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=heads, dim_feedforward=embed_dim*4, dropout=dropout, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(embed_dim // 2, NUM_CLASSES)
        )

    def forward(self, x):
        features = self.stem(x)
        features = features.permute(0, 2, 1) + self.pos_embed
        encoded = self.encoder(features)
        return self.fc(torch.mean(encoded, dim=1))

def load_base_data(folder_path, prefix):
    files = glob.glob(os.path.join(folder_path, f"{prefix}*.pt"))
    datasets = []
    for f in files:
        try:
            data = torch.load(f)
            if torch.isnan(data['X']).any(): data['X'] = torch.nan_to_num(data['X'], nan=0.0)
            if len(data['y']) > 0: datasets.append(TensorDataset(data['X'], data['y']))
        except: pass
    return ConcatDataset(datasets)

print("Veriler Yükleniyor...")
base_train = load_base_data(TRAIN_DIR, "train_part")
base_val = load_base_data(BENCHMARK_DIR, "val_part")

# Gürültü ve Ölçeklendirme SADECE Eğitim (Train) setine uygulanır! Benchmark saf kalır.
train_dataset = AugmentedECGDataset(base_train, transform=True)
val_dataset = AugmentedECGDataset(base_val, transform=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

all_y = [ds.tensors[1].numpy() for ds in base_train.datasets]
all_y = np.concatenate(all_y)
class_weights = compute_class_weight('balanced', classes=np.unique(all_y), y=all_y)
class_weights[1] *= 1.5 
weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

def load_pretrained_weights(model, pretrained_path):
    state_dict = torch.load(pretrained_path, map_location=DEVICE)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

def objective(trial):
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True) 
    
    model = Conformer_Classifier(dropout=dropout).to(DEVICE)
    model = load_pretrained_weights(model, PRETRAINED_BRAIN_PATH)
    
    criterion = FocalLoss(weight=weights_tensor, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    EPOCHS = 8 
    best_val_f1 = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            mean = inputs.mean(dim=2, keepdim=True)
            std = inputs.std(dim=2, keepdim=True)
            inputs = (inputs - mean) / (std + 1e-5)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                mean = inputs.mean(dim=2, keepdim=True)
                std = inputs.std(dim=2, keepdim=True)
                inputs = (inputs - mean) / (std + 1e-5)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
        val_f1 = f1_score(all_targets, all_preds, average='macro')
        
        trial.report(val_f1, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            temp_path = os.path.join(OUTPUT_DIR, f"finetune_aug_trial_{trial.number}_best.pth")
            torch.save(model.state_dict(), temp_path)

    return best_val_f1

if __name__ == "__main__":
    print("🚀 TEORİK SINIRLARI ZORLAYAN AUGMENTED OPTUNA BAŞLIYOR...")
    
    db_path = os.path.join(OUTPUT_DIR, "conformer_finetune_augmented.db")
    study = optuna.create_study(
        direction="maximize", 
        study_name="ekg_finetune_augmented",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=3, interval_steps=1)
    )
    
    study.optimize(objective, n_trials=15)
    
    print("\n🏆 İNCE AYAR BİTTİ! İŞTE AUGMENTED TRANSFER LEARNING ŞAMPİYONU:")
    best_trial = study.best_trial
    print(f"En Yüksek MACRO F1 Skoru: {best_trial.value:.4f}")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
        
    import shutil
    best_temp_model = os.path.join(OUTPUT_DIR, f"finetune_aug_trial_{best_trial.number}_best.pth")
    final_model_path = os.path.join(OUTPUT_DIR, "BEST_FINETUNED_AUGMENTED_CONFORMER.pth")
    if os.path.exists(best_temp_model):
        shutil.copy(best_temp_model, final_model_path)
        print(f"\n✅ Nihai Augmented Şampiyon Model kaydedildi: {final_model_path}")
        
    for f in glob.glob(os.path.join(OUTPUT_DIR, "finetune_aug_trial_*_best.pth")):
        if f != best_temp_model:
            os.remove(f)