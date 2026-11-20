import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from torchvision.models import resnet18
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
import glob
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# --- AYARLAR ---
BASE_DIR = r"C:\Users\Eymen\Desktop\EKG\GÖREV 1\düzeltilmiş_resnet_v2"
TRAIN_DIR = os.path.join(BASE_DIR, "processed_train")
BENCHMARK_DIR = os.path.join(BASE_DIR, "benchmark")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models_resnet18_std") # Farklı klasöre kaydetsin
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128 # ResNet18 hafif olduğu için yüksek batch
LR = 1e-4
EPOCHS = 20
NUM_CLASSES = 3

# --- VERİ YÜKLEME ---
def load_dataset_from_parts(folder_path, prefix):
    files = glob.glob(os.path.join(folder_path, f"{prefix}*.pt"))
    datasets = []
    print(f"Taranıyor: {folder_path} -> {len(files)} parça.")
    for f in tqdm(files):
        try:
            data = torch.load(f)
            if torch.isnan(data['X']).any(): data['X'] = torch.nan_to_num(data['X'], nan=0.0)
            if len(data['y']) > 0: datasets.append(TensorDataset(data['X'], data['y']))
        except: pass
    if not datasets: raise RuntimeError(f"{prefix} bulunamadı.")
    return ConcatDataset(datasets)

print("Veriler yükleniyor...")
train_dataset = load_dataset_from_parts(TRAIN_DIR, "train_part")
val_dataset = load_dataset_from_parts(BENCHMARK_DIR, "val_part")
test_dataset = load_dataset_from_parts(BENCHMARK_DIR, "test_part")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- AĞIRLIKLAR ---
all_y = []
for ds in train_dataset.datasets: all_y.append(ds.tensors[1].numpy())
all_y = np.concatenate(all_y)
class_weights = compute_class_weight('balanced', classes=np.unique(all_y), y=all_y)
weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
print(f"Ağırlıklar: {weights_tensor.cpu().numpy()}")

# --- MODEL: STANDART RESNET18 ---
class StandardResNet18(nn.Module):
    def __init__(self, num_classes):
        super(StandardResNet18, self).__init__()
        self.resnet = resnet18(weights=None)
        # 1 Kanal Giriş (Batch, 1, 12, 5000)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.resnet(x)

model = StandardResNet18(NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.1)

# --- EĞİTİM ---
best_f1 = 0.0
print("Eğitim Başlıyor (Standard ResNet18)...")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
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
    print(f"Epoch {epoch+1} -> Val Macro F1: {val_f1:.4f}")
    
    scheduler.step(val_f1)
    
    # --- KAYIT İŞLEMLERİ (İSTEĞİN ÜZERİNE) ---
    # 1. Her Epoch'u kaydet
    epoch_save_path = os.path.join(MODEL_SAVE_DIR, f"resnet18_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), epoch_save_path)
    
    # 2. En İyi Modeli ayrıca sakla
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "best_model.pth"))
        print(f">>> EN İYİ MODEL GÜNCELLENDİ: {val_f1:.4f} <<<")

# Test raporu buraya eklenebilir...