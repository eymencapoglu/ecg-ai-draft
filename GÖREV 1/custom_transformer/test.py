import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
from tqdm import tqdm
import warnings
import math

warnings.filterwarnings("ignore")

# --- AYARLAR ---
# Veri Kaynağı
DATA_SOURCE_DIR = r"C:\Users\Eymen\Desktop\EKG\GÖREV 1\düzeltilmiş_resnet_v2"
TEST_DATA_DIR = os.path.join(DATA_SOURCE_DIR, "benchmark")

# Modelin Yeri (Şampiyon Maraton Modeli)
MODEL_DIR = r"C:\Users\Eymen\Desktop\EKG\GÖREV 1\custom_transformer"
MODEL_PATH = os.path.join(MODEL_DIR, "FINAL_CONFORMER_MARATHON.pth")

# Sonuç Klasörü
RESULTS_DIR = os.path.join(MODEL_DIR, "test_results")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "correct_samples"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "wrong_samples"), exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
NUM_CLASSES = 3
CLASS_NAMES = ["Ritim (0)", "İletim (1)", "Normal (2)"]

# Optuna Şampiyon Parametreleri (Eğitimle birebir aynı)
BEST_EMBED_DIM = 64
BEST_NUM_HEADS = 4
BEST_NUM_LAYERS = 4
BEST_DROPOUT = 0.2925

# --- 1. MİMARİ (Eğitimdekiyle Aynı Olmak Zorunda) ---
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
        x = torch.mean(x, dim=1)
        x = self.fc(x)
        return x

# --- 2. YARDIMCI FONKSİYONLAR ---
def load_test_data(folder_path):
    files = glob.glob(os.path.join(folder_path, "test_part*.pt"))
    datasets = []
    print(f"Test verileri yükleniyor (Benchmark): {folder_path}")
    for f in tqdm(files):
        try:
            data = torch.load(f)
            if torch.isnan(data['X']).any(): data['X'] = torch.nan_to_num(data['X'], nan=0.0)
            datasets.append(TensorDataset(data['X'], data['y']))
        except: pass
    return ConcatDataset(datasets)

def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix (Conformer Model)')
    plt.ylabel('Gerçek Etiket')
    plt.xlabel('Tahmin Edilen Etiket')
    plt.savefig(save_path)
    plt.close()

def plot_ecg_sample(signal, true_lbl, pred_lbl, sample_id, save_folder):
    leads = ['Lead I', 'Lead II', 'Lead III']
    t = np.arange(signal.shape[1])
    
    is_correct = (true_lbl == pred_lbl)
    color = 'green' if is_correct else 'red'
    status_text = "DOĞRU TAHMİN" if is_correct else "YANLIŞ TAHMİN"
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    fig.suptitle(f'{status_text} (Sample #{sample_id})\nGerçek: {CLASS_NAMES[true_lbl]} | Tahmin: {CLASS_NAMES[pred_lbl]}', 
                 fontsize=16, color=color, weight='bold')

    for i in range(3):
        ax = axes[i]
        ax.plot(t, signal[i, :], label=leads[i], color='black', linewidth=0.8)
        ax.set_ylabel("mV")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc="upper right")
        if i == 0: ax.set_title(leads[i])
        
    axes[-1].set_xlabel("Zaman (Örnek)")
    filename = f"{'CORRECT' if is_correct else 'WRONG'}_{sample_id}_True_{true_lbl}_Pred_{pred_lbl}.png"
    plt.savefig(os.path.join(save_folder, filename))
    plt.close()

# --- 3. ANA TEST DÖNGÜSÜ ---
def main():
    print("--- CONFORMER MODEL TESTİ BAŞLIYOR ---")
    
    test_dataset = load_test_data(TEST_DATA_DIR)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Toplam Test Verisi: {len(test_dataset)}")

    print(f"Model yükleniyor: {MODEL_PATH}")
    model = Conformer1D(BEST_EMBED_DIM, BEST_NUM_HEADS, BEST_NUM_LAYERS, BEST_DROPOUT).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print("HATA: Model dosyası bulunamadı! Yolun doğruluğunu kontrol et.")
        return

    model.eval()
    all_preds, all_targets = [], []
    
    MAX_SAMPLES = 20 
    collected_correct = 0
    collected_wrong = 0
    
    print("Tahminler yapılıyor ve grafikler çiziliyor...")
    with torch.no_grad():
        batch_start = 0
        for inputs, targets in tqdm(test_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            # Normalizasyon
            mean = inputs.mean(dim=2, keepdim=True)
            std = inputs.std(dim=2, keepdim=True)
            inputs = (inputs - mean) / (std + 1e-5)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Örnek Kaydetme
            if collected_correct < MAX_SAMPLES or collected_wrong < MAX_SAMPLES:
                current_batch_size = inputs.size(0)
                for i in range(current_batch_size):
                    t_lbl = targets[i].item()
                    p_lbl = preds[i].item()
                    
                    if t_lbl == p_lbl and collected_correct < MAX_SAMPLES:
                        sig = inputs[i].cpu().numpy()
                        plot_ecg_sample(sig, t_lbl, p_lbl, batch_start + i, os.path.join(RESULTS_DIR, "correct_samples"))
                        collected_correct += 1
                        
                    elif t_lbl != p_lbl and collected_wrong < MAX_SAMPLES:
                        sig = inputs[i].cpu().numpy()
                        plot_ecg_sample(sig, t_lbl, p_lbl, batch_start + i, os.path.join(RESULTS_DIR, "wrong_samples"))
                        collected_wrong += 1
            
            batch_start += inputs.size(0)

    # 4. Raporlama
    print("\n--- SONUÇLAR ---")
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro')
    report = classification_report(all_targets, all_preds, target_names=CLASS_NAMES)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print("\nSınıflandırma Raporu:\n")
    print(report)
    
    with open(os.path.join(RESULTS_DIR, "conformer_test_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Model: Conformer1D (Transformer tabanlı)\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Macro F1: {f1:.4f}\n\n")
        f.write(report)

    plot_confusion_matrix(all_targets, all_preds, os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    print(f"\nGrafikler ve Rapor kaydedildi: {RESULTS_DIR}")

if __name__ == "__main__":
    main()