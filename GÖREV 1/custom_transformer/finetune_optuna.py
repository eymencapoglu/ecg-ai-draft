import torch
import torch.nn as nn
import torch.nn.functional as F
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

# --- AYARLAR ---
DATA_SOURCE_DIR = r"C:\Users\Eymen\Desktop\EKG\GÖREV 1\düzeltilmiş_resnet_v2"
TRAIN_DIR = os.path.join(DATA_SOURCE_DIR, "processed_train")
BENCHMARK_DIR = os.path.join(DATA_SOURCE_DIR, "benchmark")

OUTPUT_DIR = r"C:\Users\Eymen\Desktop\EKG\GÖREV 1\custom_transformer"
# Google Drive'dan indirdiğin dosyanın tam yolu
PRETRAINED_BRAIN_PATH = os.path.join(OUTPUT_DIR, "pretrained_ekg_brain.pth") 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
NUM_CLASSES = 3

# --- 1. ÖZEL KAYIP FONKSİYONU: FOCAL LOSS ---
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

# --- 2. MİMARİ: TRANSFER LEARNING (CONFORMER CLASSIFIER) ---
# DİKKAT: Encoder kısmı pre-training ile BİREBİR aynı olmak zorunda.
class Conformer_Classifier(nn.Module):
    def __init__(self, in_chans=12, embed_dim=128, depth=4, heads=4, dropout=0.2):
        super().__init__()
        
        # Ön Eğitimden (MAE) Gelen Kısımlar
        self.stem = nn.Sequential(
            nn.Conv1d(in_chans, embed_dim // 2, kernel_size=15, stride=5, padding=7),
            nn.BatchNorm1d(embed_dim // 2),
            nn.GELU(),
            nn.Conv1d(embed_dim // 2, embed_dim, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(embed_dim),
            nn.GELU()
        )
        self.num_patches = 250
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=heads, dim_feedforward=embed_dim*4, dropout=dropout, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # YENİ KISIM: Hastalık Sınıflandırıcı Kafa
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, NUM_CLASSES)
        )

    def forward(self, x):
        features = self.stem(x)
        features = features.permute(0, 2, 1) 
        features = features + self.pos_embed
        
        encoded = self.encoder(features)
        
        # Zaman ekseni boyunca ortalama alıp (Global Avg Pooling) tek vektöre düşürüyoruz
        pooled = torch.mean(encoded, dim=1) 
        out = self.fc(pooled)
        return out

# --- 3. VERİ YÜKLEME VE AĞIRLIKLAR ---
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

all_y = [ds.tensors[1].numpy() for ds in train_dataset.datasets]
all_y = np.concatenate(all_y)
class_weights = compute_class_weight('balanced', classes=np.unique(all_y), y=all_y)
class_weights[1] *= 1.5 # İletim Torpili
weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

# --- 4. ÖN EĞİTİM (PRETRAINED) BEYNİ YÜKLEME MANTIĞI ---
def load_pretrained_weights(model, pretrained_path):
    print(f"\n🧠 Uzman Beyin Yükleniyor: {pretrained_path}")
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError("BEYİN BULUNAMADI! Lütfen Colab'dan indirdiğin dosyayı klasöre koy.")
        
    state_dict = torch.load(pretrained_path, map_location=DEVICE)
    model_dict = model.state_dict()
    
    # Sadece isimleri ve boyutları eşleşenleri al (Böylece Decoder kısmı çöpe gider, sadece Encoder gelir)
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print(f"✅ Başarıyla transfer edilen katman sayısı: {len(pretrained_dict)} / {len(model_dict)}")
    return model

# --- 5. OPTUNA (İnce Ayar Araması) ---
def objective(trial):
    # Mimari (Katman/Genişlik) ARAMIYORUZ! Çünkü pre-trained beyin 128/4/4 formatında.
    # Sadece Classifier ve Eğitim dinamiklerini arıyoruz:
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True) # Transfer Learning'de LR genelde düşük tutulur
    
    model = Conformer_Classifier(dropout=dropout).to(DEVICE)
    model = load_pretrained_weights(model, PRETRAINED_BRAIN_PATH)
    
    criterion = FocalLoss(weight=weights_tensor, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    EPOCHS = 8 # Optuna denemeleri için 8 epoch yeterli
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
            temp_path = os.path.join(OUTPUT_DIR, f"finetune_trial_{trial.number}_best.pth")
            torch.save(model.state_dict(), temp_path)

    return best_val_f1

if __name__ == "__main__":
    print("🚀 TRANSFER LEARNING + OPTUNA BAŞLIYOR...")
    
    db_path = os.path.join(OUTPUT_DIR, "conformer_finetune.db")
    study = optuna.create_study(
        direction="maximize", 
        study_name="ekg_finetune",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=3, interval_steps=1)
    )
    
    # Pre-trained model çok hızlı adapte olur, 10-15 deneme (trial) fazlasıyla yeterlidir
    study.optimize(objective, n_trials=15)
    
    print("\n🏆 İNCE AYAR BİTTİ! İŞTE TRANSFER LEARNING ŞAMPİYONU:")
    best_trial = study.best_trial
    print(f"En Yüksek F1 Skoru: {best_trial.value:.4f}")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
        
    import shutil
    best_temp_model = os.path.join(OUTPUT_DIR, f"finetune_trial_{best_trial.number}_best.pth")
    final_model_path = os.path.join(OUTPUT_DIR, "BEST_FINETUNED_CONFORMER.pth")
    if os.path.exists(best_temp_model):
        shutil.copy(best_temp_model, final_model_path)
        print(f"\n✅ Nihai Model kaydedildi: {final_model_path}")
        
    for f in glob.glob(os.path.join(OUTPUT_DIR, "finetune_trial_*_best.pth")):
        if f != best_temp_model:
            os.remove(f)