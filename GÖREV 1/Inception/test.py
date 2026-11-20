import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
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

# --- AYARLAR ---
# Verinin olduğu yer (Benchmark klasörü)
DATA_SOURCE_DIR = r"C:\Users\Eymen\Desktop\EKG\GÖREV 1\düzeltilmiş_resnet_v2"
TEST_DATA_DIR = os.path.join(DATA_SOURCE_DIR, "benchmark")

# Modelin olduğu ve sonuçların kaydedileceği yer
MODEL_DIR = r"C:\Users\Eymen\Desktop\EKG\GÖREV 1\Inception"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")

# Sonuç Klasörü
RESULTS_DIR = os.path.join(MODEL_DIR, "test_results")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "correct_samples"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "wrong_samples"), exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
NUM_CLASSES = 3
CLASS_NAMES = ["Ritim (0)", "İletim (1)", "Normal (2)"]

# --- 1. MODEL MİMARİSİ (Eğitimdeki ile BİREBİR AYNI OLMALI) ---

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.query_layer = nn.Linear(hidden_dim, 256)
        self.value_layer = nn.Linear(hidden_dim, 256)
        self.v = nn.Linear(256, 1, bias=False)

    def forward(self, x):
        q = torch.tanh(self.query_layer(x))
        scores = self.v(q)
        weights = F.softmax(scores, dim=1)
        return x * weights

class TeknofestAttentionModel(nn.Module):
    def __init__(self, num_classes):
        super(TeknofestAttentionModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=12, out_channels=128, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_dim=256)
        self.fc1 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.4)
        self.fc_out = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.attention(x)
        avg_pool = torch.mean(x, dim=1)
        max_pool, _ = torch.max(x, dim=1)
        combined = torch.cat([avg_pool, max_pool], dim=1)
        x = self.fc1(combined)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_out(x)
        return x

# --- 2. YARDIMCI FONKSİYONLAR ---

def load_test_data(folder_path):
    files = glob.glob(os.path.join(folder_path, "test_part*.pt"))
    datasets = []
    print(f"Test verileri yükleniyor: {folder_path} -> {len(files)} parça.")
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
    plt.title('Confusion Matrix (Teknofest Model)')
    plt.ylabel('Gerçek Etiket')
    plt.xlabel('Tahmin Edilen Etiket')
    plt.savefig(save_path)
    plt.close()

def plot_ecg_sample(signal, true_lbl, pred_lbl, sample_id, save_folder):
    leads = ['Lead I', 'Lead II', 'Lead III']
    # Sinyal (12, 5000). İlk 3 kanalı çiziyoruz.
    t = np.arange(signal.shape[1])
    
    # Renk ayarı: Doğruysa Yeşil, Yanlışsa Kırmızı
    is_correct = (true_lbl == pred_lbl)
    color = 'green' if is_correct else 'red'
    status_text = "DOĞRU TAHMİN" if is_correct else "YANLIŞ TAHMİN"
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    
    # Ana Başlık
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
    print("--- TEKNOFEST MODEL TESTİ BAŞLIYOR ---")
    
    # 1. Veri Yükle
    test_dataset = load_test_data(TEST_DATA_DIR)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Toplam Test Verisi: {len(test_dataset)}")

    # 2. Modeli Yükle
    print(f"Model yükleniyor: {MODEL_PATH}")
    model = TeknofestAttentionModel(NUM_CLASSES).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print("HATA: Model dosyası bulunamadı! Önce eğitimi tamamla.")
        return

    model.eval()

    # 3. Tahmin
    all_preds = []
    all_targets = []
    
    # Görselleştirme için havuz
    correct_indices = []
    wrong_indices = []
    
    # Sinyalleri hafızada tutmak yerine indeksleri tutacağız
    # Ancak PyTorch DataLoader shuffle=False olduğu için indeks sırası aynıdır.
    # Yine de örnek çizmek için batch içindeki sinyali o an kaydetmek en kolayı.
    
    # Örnek toplama limitleri
    MAX_SAMPLES = 20 
    collected_correct = 0
    collected_wrong = 0
    
    print("Tahminler yapılıyor...")
    
    with torch.no_grad():
        batch_start = 0
        for inputs, targets in tqdm(test_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            # Normalizasyon (Eğitimdekiyle aynı)
            mean = inputs.mean(dim=2, keepdim=True)
            std = inputs.std(dim=2, keepdim=True)
            inputs = (inputs - mean) / (std + 1e-5)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # --- Örnek Kaydetme Mantığı ---
            # Sadece belirli sayıda örnek kaydedene kadar çalışır
            if collected_correct < MAX_SAMPLES or collected_wrong < MAX_SAMPLES:
                current_batch_size = inputs.size(0)
                for i in range(current_batch_size):
                    t_lbl = targets[i].item()
                    p_lbl = preds[i].item()
                    
                    if t_lbl == p_lbl and collected_correct < MAX_SAMPLES:
                        # Doğru tahmin örneği
                        sig = inputs[i].cpu().numpy()
                        plot_ecg_sample(sig, t_lbl, p_lbl, batch_start + i, os.path.join(RESULTS_DIR, "correct_samples"))
                        collected_correct += 1
                        
                    elif t_lbl != p_lbl and collected_wrong < MAX_SAMPLES:
                        # Yanlış tahmin örneği
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
    
    # Raporu kaydet
    with open(os.path.join(RESULTS_DIR, "final_test_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Model: Teknofest Attention (Inception)\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Macro F1: {f1:.4f}\n\n")
        f.write(report)

    # Confusion Matrix
    plot_confusion_matrix(all_targets, all_preds, os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    print(f"Grafikler ve Rapor kaydedildi: {RESULTS_DIR}")

if __name__ == "__main__":
    main()