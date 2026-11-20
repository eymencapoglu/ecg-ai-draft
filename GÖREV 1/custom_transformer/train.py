import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
import glob
from tqdm import tqdm
import warnings
import math
import optuna
from optuna.trial import TrialState

warnings.filterwarnings("ignore")

# --- KLASÖR AYARLARI ---
# Veri Kaynağı (Eski klasörden alıyoruz, kopyalamaya gerek yok)
DATA_SOURCE_DIR = r"C:\Users\Eymen\Desktop\EKG\GÖREV 1\düzeltilmiş_resnet_v2"
TRAIN_DIR = os.path.join(DATA_SOURCE_DIR, "processed_train")
BENCHMARK_DIR = os.path.join(DATA_SOURCE_DIR, "benchmark")

# YENİ ÇALIŞMA ALANI (Her şey buraya kaydedilecek)
OUTPUT_DIR = r"C:\Users\Eymen\Desktop\EKG\GÖREV 1\custom_transformer"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
NUM_CLASSES = 3

# --- 1. MİMARİ: 1D CONFORMER (CNN + TRANSFORMER + GELU) ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # (1, max_len, d_model)

    def forward(self, x):
        # x: (Batch, Seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x

class Conformer1D(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, num_layers=2, dropout=0.2):
        super(Conformer1D, self).__init__()
        
        # 1. CNN Tokenizer (Sinyali 5000'den 250'ye düşürür)
        # Giriş: (B, 12, 5000)
        self.stem = nn.Sequential(
            nn.Conv1d(12, embed_dim // 2, kernel_size=15, stride=5, padding=7), # -> (B, dim/2, 1000)
            nn.BatchNorm1d(embed_dim // 2),
            nn.GELU(),
            nn.Conv1d(embed_dim // 2, embed_dim, kernel_size=7, stride=4, padding=3), # -> (B, dim, 250)
            nn.BatchNorm1d(embed_dim),
            nn.GELU()
        )
        
        # 2. Transformer Encoder
        self.pos_encoder = PositionalEncoding(d_model=embed_dim, max_len=500)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4, 
            dropout=dropout, 
            activation='gelu', # ŞAMPİYONLAR LİGİ AKTİVASYONU
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Classifier
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, NUM_CLASSES)
        )

    def forward(self, x):
        # x: (Batch, 12, 5000)
        x = self.stem(x) # -> (Batch, embed_dim, 250)
        x = x.permute(0, 2, 1) # -> (Batch, 250, embed_dim) Transformer sıralaması
        
        x = self.pos_encoder(x)
        x = self.transformer(x) # -> (Batch, 250, embed_dim)
        
        # Global Average Pooling (Zaman ekseninde ortalama al)
        x = torch.mean(x, dim=1) # -> (Batch, embed_dim)
        
        x = self.fc(x)
        return x

# --- 2. VERİ YÜKLEME ---
def load_data(folder_path, prefix):
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
train_dataset = load_data(TRAIN_DIR, "train_part")
val_dataset = load_data(BENCHMARK_DIR, "val_part")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Ağırlık Hesaplama (İletim için yine 1.5 torpil yapıyoruz)
all_y = [ds.tensors[1].numpy() for ds in train_dataset.datasets]
all_y = np.concatenate(all_y)
class_weights = compute_class_weight('balanced', classes=np.unique(all_y), y=all_y)
class_weights[1] *= 1.5 
weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

# --- 3. OPTUNA (NAS & HIPERPARAMETRE ARAMA) ---

def objective(trial):
    # Optuna'nın deneyeceği hiperparametre uzayı:
    embed_dim = trial.suggest_categorical("embed_dim", [64, 128, 256])
    num_heads = trial.suggest_categorical("num_heads", [4, 8])
    num_layers = trial.suggest_int("num_layers", 1, 4)
    dropout = trial.suggest_float("dropout", 0.1, 0.4)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    
    # Model Kur
    model = Conformer1D(embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers, dropout=dropout).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4) # Transformer'lar AdamW sever
    
    EPOCHS = 10 # Her deneme için 10 Epoch (Hızlı arama için)
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
            
        # Validation
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
        
        # Optuna Pruning (Erken Kesme) Bildirimi
        trial.report(val_f1, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            # Bu denemenin (trial) en iyi modelini geçici olarak kaydet
            temp_path = os.path.join(OUTPUT_DIR, f"trial_{trial.number}_best.pth")
            torch.save(model.state_dict(), temp_path)

    return best_val_f1

# --- 4. ARAMAYI BAŞLAT ---
if __name__ == "__main__":
    print("🚀 OPTUNA MİMARİ ARAMASI BAŞLIYOR (Conformer1D)...")
    
    # Optuna veritabanını yeni klasörüne kaydet
    db_path = os.path.join(OUTPUT_DIR, "conformer_nas.db")
    study = optuna.create_study(
        direction="maximize", 
        study_name="ekg_conformer",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=3, interval_steps=1)
    )
    
    # N_TRIALS: Kaç farklı mimari denesin? Başlangıç için 10-15 yeterlidir.
    study.optimize(objective, n_trials=15)
    
    print("\n🏆 ARAMA BİTTİ! İŞTE EN MÜKEMMEL MİMARİ:")
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    
    print(f"Budanan (Kötü) Denemeler: {len(pruned_trials)}")
    print(f"Tamamlanan Denemeler: {len(complete_trials)}")
    
    best_trial = study.best_trial
    print(f"En Yüksek F1 Skoru: {best_trial.value:.4f}")
    print("En İyi Hiperparametreler (Bu değerleri raporunda kesinlikle paylaş!):")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
        
    # En iyi modeli "Nihai Şampiyon" olarak kopyala
    import shutil
    best_temp_model = os.path.join(OUTPUT_DIR, f"trial_{best_trial.number}_best.pth")
    final_model_path = os.path.join(OUTPUT_DIR, "BEST_CONFORMER_MODEL.pth")
    if os.path.exists(best_temp_model):
        shutil.copy(best_temp_model, final_model_path)
        print(f"\n✅ Şampiyon Model kaydedildi: {final_model_path}")
        
    # Temizlik (Diğer trial_... .pth dosyalarını silebilirsin yer kaplamasın diye)
    for f in glob.glob(os.path.join(OUTPUT_DIR, "trial_*_best.pth")):
        if f != best_temp_model:
            os.remove(f)