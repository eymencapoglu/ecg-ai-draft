import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
import glob
from tqdm import tqdm
import warnings
import math

warnings.filterwarnings("ignore")

# --- AYARLAR ---
DATA_SOURCE_DIR = r"C:\Users\Eymen\Desktop\EKG\GÖREV 1\düzeltilmiş_resnet_v2"
TRAIN_DIR = os.path.join(DATA_SOURCE_DIR, "processed_train")
BENCHMARK_DIR = os.path.join(DATA_SOURCE_DIR, "benchmark")

OUTPUT_DIR = r"C:\Users\Eymen\Desktop\EKG\GÖREV 1\custom_transformer"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
NUM_CLASSES = 3

# --- OPTUNA'NIN BULDUĞU ŞAMPİYON PARAMETRELER ---
BEST_EMBED_DIM = 64
BEST_NUM_HEADS = 4
BEST_NUM_LAYERS = 4
BEST_DROPOUT = 0.2925
BEST_LR = 0.000839
EPOCHS = 40 # Maraton Koşusu

# --- 1. ÖZEL KAYIP FONKSİYONU: FOCAL LOSS ---
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.weight = weight # Sınıf ağırlıkları (Class-Balanced)
        self.gamma = gamma   # Odaklanma şiddeti (Genelde 2.0 idealdir)

    def forward(self, inputs, targets):
        # 1. Standart Cross Entropy'yi hesapla (weight ile)
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        
        # 2. Olasılık (pt) değerini bul
        pt = torch.exp(-ce_loss)
        
        # 3. Focal Loss formülü: (1 - pt)^gamma * CE_Loss
        # Eğer model eminse (pt yüksekse), (1-pt) sıfıra yaklaşır, loss düşer.
        # Eğer model zorlanıyorsa (pt düşükse), ceza katlanarak artar.
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        return focal_loss.mean()

# --- 2. MİMARİ (CONFORMER) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

class Conformer1D(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, dropout):
        super(Conformer1D, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(12, embed_dim // 2, kernel_size=15, stride=5, padding=7),
            nn.BatchNorm1d(embed_dim // 2),
            nn.GELU(),
            nn.Conv1d(embed_dim // 2, embed_dim, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(embed_dim),
            nn.GELU()
        )
        self.pos_encoder = PositionalEncoding(d_model=embed_dim, max_len=500)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, 
            dim_feedforward=embed_dim * 4, dropout=dropout, 
            activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.stem(x)
        x = x.permute(0, 2, 1)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = torch.mean(x, dim=1) # Global Average Pooling
        x = self.fc(x)
        return x

# --- 3. VERİ YÜKLEME VE EĞİTİM HAZIRLIĞI ---
def load_data(folder_path, prefix):
    files = glob.glob(os.path.join(folder_path, f"{prefix}*.pt"))
    datasets = []
    print(f"[{prefix}] Verileri Yükleniyor...")
    for f in tqdm(files):
        try:
            data = torch.load(f)
            if torch.isnan(data['X']).any(): data['X'] = torch.nan_to_num(data['X'], nan=0.0)
            if len(data['y']) > 0: datasets.append(TensorDataset(data['X'], data['y']))
        except: pass
    return ConcatDataset(datasets)

train_dataset = load_data(TRAIN_DIR, "train_part")
val_dataset = load_data(BENCHMARK_DIR, "val_part")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Sınıf Ağırlıkları
print("\nSınıf ağırlıkları hesaplanıyor...")
all_y = [ds.tensors[1].numpy() for ds in train_dataset.datasets]
all_y = np.concatenate(all_y)
class_weights = compute_class_weight('balanced', classes=np.unique(all_y), y=all_y)
class_weights[1] *= 1.5 # İletim Torpili
weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
print(f"Kullanılacak Ağırlıklar: {class_weights}")

# Model, Loss ve Optimizer Kurulumu
model = Conformer1D(BEST_EMBED_DIM, BEST_NUM_HEADS, BEST_NUM_LAYERS, BEST_DROPOUT).to(DEVICE)

# YENİLİK 1: Focal Loss kullanıyoruz
criterion = FocalLoss(weight=weights_tensor, gamma=2.0)

optimizer = optim.AdamW(model.parameters(), lr=BEST_LR, weight_decay=1e-4)

# YENİLİK 2: Cosine Annealing (Öğrenme Oranını dalga gibi soğutur, takılmaları önler)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# --- 4. MARATON EĞİTİM DÖNGÜSÜ ---
best_f1 = 0.0
print("\n🚀 MARATON BAŞLIYOR: CONFORMER + FOCAL LOSS (40 EPOCH)")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
    for inputs, targets in loop:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        # Normalizasyon
        mean = inputs.mean(dim=2, keepdim=True)
        std = inputs.std(dim=2, keepdim=True)
        inputs = (inputs - mean) / (std + 1e-5)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Gradient Clipping (Patlamaları engeller)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

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
    val_acc = accuracy_score(all_targets, all_preds)
    current_lr = scheduler.get_last_lr()[0]
    
    print(f"Epoch {epoch+1} Özet -> Val Macro F1: {val_f1:.4f} | Acc: {val_acc:.4f} | LR: {current_lr:.6f}")
    
    # Cosine Annealing adımını at
    scheduler.step()
    
    if val_f1 > best_f1:
        best_f1 = val_f1
        save_path = os.path.join(OUTPUT_DIR, "FINAL_CONFORMER_MARATHON.pth")
        torch.save(model.state_dict(), save_path)
        print(f"🏆 >>> YENİ REKOR! Şampiyon Model Kaydedildi: {val_f1:.4f} <<<")

print(f"\n✅ Eğitim Tamamlandı. En İyi Macro F1: {best_f1:.4f}")
print("Nihai Rapor:")
print(classification_report(all_targets, all_preds, target_names=["Ritim", "İletim", "Normal"]))