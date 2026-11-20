import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from torchvision.models import resnet18, resnet101
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import glob
from tqdm import tqdm
import warnings

# Gereksiz uyarıları kapat
warnings.filterwarnings("ignore")

# --- KULLANICI AYARLARI ---
# Hangi modeli test etmek istiyorsun? "standard" veya "hybrid" yaz.
MODEL_TYPE = "hybrid"  
# MODEL_TYPE = "standard"

# Dosya Yolları
BASE_DIR = r"C:\Users\Eymen\Desktop\EKG\GÖREV 1\düzeltilmiş_resnet_v2"
BENCHMARK_DIR = os.path.join(BASE_DIR, "benchmark")

# Modelin kayıtlı olduğu klasör (Eğitimde belirlediğin yollar)
if MODEL_TYPE == "hybrid":
    MODEL_PATH = os.path.join(BASE_DIR, "models_hybrid_final", "best_model.pth")
else:
    MODEL_PATH = os.path.join(BASE_DIR, "models_resnet18_std", "best_model.pth")

# Sonuçların kaydedileceği klasör
RESULTS_DIR = os.path.join(BASE_DIR, f"test_results_{MODEL_TYPE}")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "sample_plots"), exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
NUM_CLASSES = 3
CLASS_NAMES = ["Ritim (0)", "İletim (1)", "Normal (2)"]

# --- MODEL TANIMLAMALARI (Eğitim kodundan alındı) ---

class SEBlock(nn.Module):
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
    def __init__(self, feature_dim, step_dim, bias=True):
        super(Attention, self).__init__()
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.bias = bias
        self.param_weight = nn.Parameter(torch.Tensor(feature_dim, 1))
        self.param_bias = nn.Parameter(torch.Tensor(1))
        self.soft_max = nn.Softmax(dim=1)
        nn.init.xavier_uniform_(self.param_weight)
        if self.bias: self.param_bias.data.fill_(0)

    def forward(self, x):
        eij = torch.mm(x.contiguous().view(-1, self.feature_dim), self.param_weight).view(-1, self.step_dim)
        if self.bias: eij = eij + self.param_bias
        eij = torch.tanh(eij)
        a = self.soft_max(eij)
        weighted_input = x * a.unsqueeze(2).expand_as(x)
        return torch.sum(weighted_input, 1)

class HybridResNet18(nn.Module):
    def __init__(self, num_classes):
        super(HybridResNet18, self).__init__()
        self.resnet = resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.encoder = nn.Sequential(*list(self.resnet.children())[:-2])
        self.se_block = SEBlock(512)
        self.hidden_dim = 128
        self.lstm = nn.LSTM(input_size=512, hidden_size=self.hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.attention = Attention(feature_dim=self.hidden_dim * 2, step_dim=157)
        self.fc = nn.Sequential(nn.Linear(self.hidden_dim * 2, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, num_classes))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.se_block(x)
        x = x.squeeze(2).permute(0, 2, 1)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x = self.attention(x)
        x = self.fc(x)
        return x

class StandardResNet18(nn.Module):
    def __init__(self, num_classes):
        super(StandardResNet18, self).__init__()
        self.resnet = resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    def forward(self, x):
        x = x.unsqueeze(1)
        return self.resnet(x)

# --- YARDIMCI FONKSİYONLAR ---

def load_test_data(folder_path):
    files = glob.glob(os.path.join(folder_path, "test_part*.pt"))
    datasets = []
    print(f"Test verileri yükleniyor: {folder_path} -> {len(files)} parça.")
    for f in tqdm(files):
        try:
            data = torch.load(f)
            # Normalizasyon veya NaN temizliği gerekirse burada yapılabilir
            if torch.isnan(data['X']).any(): data['X'] = torch.nan_to_num(data['X'], nan=0.0)
            datasets.append(TensorDataset(data['X'], data['y']))
        except: pass
    return ConcatDataset(datasets)

def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'Confusion Matrix ({MODEL_TYPE})')
    plt.ylabel('Gerçek Etiket')
    plt.xlabel('Tahmin Edilen Etiket')
    plt.savefig(save_path)
    plt.close()

def plot_ecg_sample(signal, true_label, pred_label, sample_id, save_dir):
    # Sinyal: (12, 5000) -> Biz ilk 3 kanalı çizelim (Lead I, II, III varsayımı)
    leads = ['Lead I', 'Lead II', 'Lead III']
    t = np.arange(signal.shape[1])
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    color = 'green' if true_label == pred_label else 'red'
    
    fig.suptitle(f'Sample #{sample_id}\nGerçek: {CLASS_NAMES[true_label]} | Tahmin: {CLASS_NAMES[pred_label]}', 
                 fontsize=16, color=color, weight='bold')

    for i in range(3):
        ax = axes[i]
        ax.plot(t, signal[i, :], label=leads[i])
        ax.set_ylabel("mV")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc="upper right")
        
    axes[-1].set_xlabel("Zaman (Sample)")
    
    save_path = os.path.join(save_dir, f"sample_{sample_id}_True_{true_label}_Pred_{pred_label}.png")
    plt.savefig(save_path)
    plt.close()

# --- ANA TEST AKIŞI ---

def main():
    print(f"--- {MODEL_TYPE.upper()} MODEL TESTİ BAŞLIYOR ---")
    
    # 1. Veri Yükle
    test_dataset = load_test_data(BENCHMARK_DIR)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Toplam Test Verisi: {len(test_dataset)}")

    # 2. Modeli Yükle
    print(f"Model yükleniyor: {MODEL_PATH}")
    if MODEL_TYPE == "hybrid":
        model = HybridResNet18(NUM_CLASSES).to(DEVICE)
    else:
        model = StandardResNet18(NUM_CLASSES).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print("HATA: Model dosyası bulunamadı! Lütfen önce eğitimi tamamlayın.")
        return

    model.eval()

    # 3. Tahmin Döngüsü
    all_preds = []
    all_targets = []
    
    # Görselleştirme için rastgele örnekler topla (Örn: 20 tane)
    plot_indices = np.random.choice(len(test_dataset), 20, replace=False)
    plot_counter = 0

    print("Tahminler yapılıyor...")
    with torch.no_grad():
        # Batch batch ilerle
        batch_start_idx = 0
        for inputs, targets in tqdm(test_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            # Test Normalizasyonu (Eğitimdeki gibi)
            mean = inputs.mean(dim=2, keepdim=True)
            std = inputs.std(dim=2, keepdim=True)
            inputs = (inputs - mean) / (std + 1e-5)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # -- Görselleştirme Kısmı --
            # Bu batch'in içindeki indeksleri kontrol et
            batch_size = inputs.size(0)
            for i in range(batch_size):
                global_idx = batch_start_idx + i
                if global_idx in plot_indices:
                    # CPU'ya al ve çiz
                    sig = inputs[i].cpu().numpy()
                    t_lbl = targets[i].item()
                    p_lbl = preds[i].item()
                    plot_ecg_sample(sig, t_lbl, p_lbl, global_idx, os.path.join(RESULTS_DIR, "sample_plots"))
            
            batch_start_idx += batch_size

    # 4. Raporlama ve Kayıt
    print("\n--- SONUÇLAR HESAPLANIYOR ---")
    
    # A. Metrikler
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro')
    report = classification_report(all_targets, all_preds, target_names=CLASS_NAMES)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print("\nSınıflandırma Raporu:\n")
    print(report)
    
    # Raporu Text olarak kaydet
    with open(os.path.join(RESULTS_DIR, "final_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Model: {MODEL_TYPE}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Macro F1: {f1:.4f}\n\n")
        f.write(report)

    # B. Confusion Matrix
    plot_confusion_matrix(all_targets, all_preds, os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    print(f"Confusion Matrix kaydedildi: {RESULTS_DIR}\\confusion_matrix.png")

    # C. Tüm Sonuçları CSV Yap
    df = pd.DataFrame({
        'Sample_ID': range(len(all_targets)),
        'True_Label': all_targets,
        'Pred_Label': all_preds,
        'True_Class': [CLASS_NAMES[t] for t in all_targets],
        'Pred_Class': [CLASS_NAMES[p] for p in all_preds]
    })
    df.to_csv(os.path.join(RESULTS_DIR, "all_predictions.csv"), index=False)
    print(f"Tüm tahminler kaydedildi: {RESULTS_DIR}\\all_predictions.csv")
    print(f"Örnek grafikler kaydedildi: {RESULTS_DIR}\\sample_plots")

if __name__ == "__main__":
    main()