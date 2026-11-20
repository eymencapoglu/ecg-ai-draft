import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from torchvision.models import resnet101
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
import glob
from tqdm import tqdm
import warnings

# Gereksiz uyarıları kapat
warnings.filterwarnings("ignore")

# --- AYARLAR ---
BASE_DIR = r"C:\Users\Eymen\Desktop\EKG\GÖREV 1\düzeltilmiş_resnet_v2"
TRAIN_DIR = os.path.join(BASE_DIR, "processed_train")
BENCHMARK_DIR = os.path.join(BASE_DIR, "benchmark")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ResNet101 + LSTM ağırdır. OOM (Out of Memory) hatası alırsan bunu düşür.
BATCH_SIZE = 64 
LR = 1e-4 
EPOCHS = 20 
NUM_CLASSES = 3

# --- 1. MODEL BİLEŞENLERİ (GELİŞMİŞ MİMARİ) ---

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block: Kanal önemini belirler."""
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x): 
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Attention(nn.Module):
    """Attention Mechanism: Sinyalin önemli zaman aralıklarına odaklanır."""
    def __init__(self, feature_dim, step_dim, bias=True):
        super(Attention, self).__init__()
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.bias = bias
        self.param_weight = nn.Parameter(torch.Tensor(feature_dim, 1))
        self.param_bias = nn.Parameter(torch.Tensor(1))
        self.soft_max = nn.Softmax(dim=1)
        
        nn.init.xavier_uniform_(self.param_weight)
        if self.bias:
            self.param_bias.data.fill_(0)

    def forward(self, x):
        # x: (Batch, Time, Features)
        eij = torch.mm(
            x.contiguous().view(-1, self.feature_dim), 
            self.param_weight
        ).view(-1, self.step_dim)
        
        if self.bias:
            eij = eij + self.param_bias
            
        eij = torch.tanh(eij)
        a = self.soft_max(eij) 
        
        weighted_input = x * a.unsqueeze(2).expand_as(x)
        output = torch.sum(weighted_input, 1)
        return output

class AdvancedECGModel(nn.Module):
    def __init__(self, num_classes):
        super(AdvancedECGModel, self).__init__()
        
        # A. ResNet101 Backbone
        self.resnet = resnet101(weights=None)
        # 12 Kanal Giriş
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # Son katmanları at, sadece özellik çıkarıcıyı al
        self.encoder = nn.Sequential(*list(self.resnet.children())[:-2])
        
        # B. SE Block (Kanal Dikkati)
        self.se_block = SEBlock(2048) # ResNet101 çıkışı 2048 kanaldır

        # C. Bi-LSTM (Zamansal İlişki)
        self.hidden_dim = 256
        self.lstm = nn.LSTM(
            input_size=2048, 
            hidden_size=self.hidden_dim, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.2
        )
        
        # D. Attention (Odaklanma)
        # Not: 5000 uzunluğundaki sinyal ResNet sonrası yaklaşık 157 uzunluğuna düşer.
        self.attention = Attention(feature_dim=self.hidden_dim * 2, step_dim=157) 
        
        # E. Classifier
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (Batch, 12, 5000) -> ResNet Input: (Batch, 1, 12, 5000)
        x = x.unsqueeze(1)
        x = self.encoder(x) # Çıktı: (Batch, 2048, 1, 157)
        
        # SE Block
        x = self.se_block(x)
        
        # Boyut Düzenleme: (Batch, Time, Features) formatına çevir
        x = x.squeeze(2).permute(0, 2, 1) # (Batch, 157, 2048)
        
        # LSTM
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x) # (Batch, 157, 512)
        
        # Attention
        x = self.attention(x) # (Batch, 512)
        
        # Sınıflandırma
        x = self.fc(x)
        return x

# --- 2. VERİ YÜKLEME ---

def load_dataset_from_parts(folder_path, prefix):
    files = glob.glob(os.path.join(folder_path, f"{prefix}*.pt"))
    datasets = []
    print(f"Konum taranıyor: {folder_path} -> {len(files)} parça bulundu.")
    
    for f in tqdm(files, desc=f"{prefix} yükleniyor"):
        try:
            data = torch.load(f)
            # Veri içinde NaN temizliği
            if torch.isnan(data['X']).any():
                data['X'] = torch.nan_to_num(data['X'], nan=0.0)
                
            if len(data['y']) > 0:
                datasets.append(TensorDataset(data['X'], data['y']))
        except Exception as e:
            print(f"\nUYARI: '{os.path.basename(f)}' okunamadı. Atlanıyor.")
            
    if not datasets:
        raise RuntimeError(f"Hata: {folder_path} içinde veri bulunamadı!")
        
    return ConcatDataset(datasets)

print("Datasetler yükleniyor...")
try:
    train_dataset = load_dataset_from_parts(TRAIN_DIR, "train_part")
    val_dataset = load_dataset_from_parts(BENCHMARK_DIR, "val_part")
    test_dataset = load_dataset_from_parts(BENCHMARK_DIR, "test_part")
except Exception as e:
    print(f"KRİTİK HATA: {e}")
    exit()

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"\n>>> Veri Durumu: Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

# --- 3. AĞIRLIK HESAPLAMA & MODEL KURULUMU ---

print("Sınıf ağırlıkları hesaplanıyor...")
all_train_labels = []
for ds in train_dataset.datasets:
    all_train_labels.append(ds.tensors[1].numpy())

all_train_labels = np.concatenate(all_train_labels)
class_weights = compute_class_weight(
    class_weight='balanced', 
    classes=np.unique(all_train_labels), 
    y=all_train_labels
)
weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
print(f">>> Hesaplanan Ağırlıklar: {weights_tensor.cpu().numpy()}") 

# Model Başlatma
model = AdvancedECGModel(num_classes=NUM_CLASSES).to(DEVICE)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.1)

# --- 4. EĞİTİM DÖNGÜSÜ ---

best_f1 = 0.0
print(f"\nEğitim Başlıyor ({EPOCHS} Epoch) - Model: ResNet101 + BiLSTM + Attention...")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for inputs, targets in loop:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        # --- NORMALİZASYON ---
        mean = inputs.mean(dim=2, keepdim=True)
        std = inputs.std(dim=2, keepdim=True)
        inputs = (inputs - mean) / (std + 1e-5)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        if torch.isnan(loss):
            continue 

        loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    # --- VALIDATION ---
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            # Val Normalizasyonu
            mean = inputs.mean(dim=2, keepdim=True)
            std = inputs.std(dim=2, keepdim=True)
            inputs = (inputs - mean) / (std + 1e-5)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    # Metrikler
    epoch_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0
    val_f1 = f1_score(all_targets, all_preds, average='macro')
    val_acc = accuracy_score(all_targets, all_preds)
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"Epoch {epoch+1} Sonuç -> Loss: {epoch_loss:.4f} | Val Macro F1: {val_f1:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr}")
    
    scheduler.step(val_f1)
    
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "best_model.pth"))
        print(f">>> EN İYİ MODEL KAYDEDİLDİ: {val_f1:.4f} <<<")

# --- 5. FINAL TEST ---
print("\n--- TEST AŞAMASI (BENCHMARK) ---")
if os.path.exists(os.path.join(MODEL_SAVE_DIR, "best_model.pth")):
    model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_DIR, "best_model.pth")))
    print("En iyi model yüklendi.")

model.eval()
test_preds, test_targets = [], []
with torch.no_grad():
    for inputs, targets in tqdm(test_loader, desc="Test ediliyor"):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        # Test Normalizasyonu
        mean = inputs.mean(dim=2, keepdim=True)
        std = inputs.std(dim=2, keepdim=True)
        inputs = (inputs - mean) / (std + 1e-5)
        
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        test_preds.extend(preds.cpu().numpy())
        test_targets.extend(targets.cpu().numpy())

print("\n" + "="*40)
print("FINAL PERFORMANS RAPORU")
print("="*40)
target_names = ["Ritim (0)", "İletim (1)", "Normal (2)"]

# Sınıf bazlı detaylı rapor
print(classification_report(test_targets, test_preds, target_names=target_names, zero_division=0))

# Özet Metrikler
final_acc = accuracy_score(test_targets, test_preds)
final_f1 = f1_score(test_targets, test_preds, average='macro')

print(f"FINAL TEST ACCURACY: {final_acc:.4f}")
print(f"FINAL TEST MACRO F1: {final_f1:.4f}")
print("="*40)