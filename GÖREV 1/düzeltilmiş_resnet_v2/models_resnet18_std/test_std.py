import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from torchvision.models import resnet18
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# --- AYARLAR ---
BASE_DIR = r"C:\Users\Eymen\Desktop\EKG\GÖREV 1\düzeltilmiş_resnet_v2"
TEST_DATA_DIR = os.path.join(BASE_DIR, "benchmark")

# Modelin Yeri (Kendi klasörüne göre ayarla)
MODEL_DIR = r"c:\Users\Eymen\Desktop\EKG\GÖREV 1\düzeltilmiş_resnet_v2\models_resnet18_std"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")

# Sonuç Klasörü
RESULTS_DIR = os.path.join(MODEL_DIR, "test_results_advanced")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "correct_samples"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "wrong_samples"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "suspicious_samples"), exist_ok=True) 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
NUM_CLASSES = 3
CLASS_NAMES = ["Ritim (0)", "İletim (1)", "Normal (2)"]

# --- 1. MODEL MİMARİSİ (Senin 0.85 aldığın Orijinal Saf ResNet18) ---
class OriginalResNet18ECG(nn.Module):
    def __init__(self, num_classes=3):
        super(OriginalResNet18ECG, self).__init__()
        self.resnet = resnet18(weights=None)
        
        # 1 kanallı girişe (EKG) uygun hale getiriyoruz
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # Orijinal eğitimde fc katmanını 3 sınıfa düşürmüştün
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1) # [Batch, 1, Channels, Length]
        return self.resnet(x)

# --- 2. YARDIMCI FONKSİYONLAR ---
def load_test_data(folder_path):
    files = glob.glob(os.path.join(folder_path, "test_part*.pt"))
    datasets = []
    print(f"Test verileri yükleniyor: {len(files)} parça bulundu.")
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
    plt.title('Confusion Matrix (Orijinal ResNet18)')
    plt.ylabel('Gerçek Etiket')
    plt.xlabel('Tahmin Edilen Etiket')
    plt.savefig(save_path)
    plt.close()

def plot_ecg_sample(signal, true_lbl, probs, pred_lbl, sample_id, save_folder, is_suspicious=False):
    leads = ['Lead I', 'Lead II', 'Lead III']
    t = np.arange(signal.shape[1])
    
    top2_prob, top2_idx = torch.topk(probs, 2)
    
    if is_suspicious:
        color = 'purple'
        status_text = "ŞÜPHELİ / ÇİFT TANILI"
        title_text = f"{status_text} (Sample #{sample_id})\nGerçek: {CLASS_NAMES[true_lbl]} | Model Kafası Karışık!\n"
        title_text += f"1. İhtimal ({CLASS_NAMES[top2_idx[0]]}): %{top2_prob[0]*100:.1f} | 2. İhtimal ({CLASS_NAMES[top2_idx[1]]}): %{top2_prob[1]*100:.1f}"
        filename_prefix = "SUSPICIOUS"
    else:
        is_correct = (true_lbl == pred_lbl)
        color = 'green' if is_correct else 'red'
        status_text = "DOĞRU TAHMİN" if is_correct else "YANLIŞ TAHMİN"
        title_text = f"{status_text} (Sample #{sample_id})\nGerçek: {CLASS_NAMES[true_lbl]} | Tahmin: {CLASS_NAMES[pred_lbl]} (%{top2_prob[0]*100:.1f})"
        filename_prefix = "CORRECT" if is_correct else "WRONG"

    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    fig.suptitle(title_text, fontsize=14, color=color, weight='bold')

    for i in range(3):
        ax = axes[i]
        ax.plot(t, signal[i, :], label=leads[i], color='black', linewidth=0.8)
        ax.set_ylabel("mV")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc="upper right")
        
    axes[-1].set_xlabel("Zaman (Örnek)")
    filename = f"{filename_prefix}_{sample_id}_T_{true_lbl}_P_{pred_lbl}.png"
    plt.savefig(os.path.join(save_folder, filename))
    plt.close()

# --- 3. ANA TEST DÖNGÜSÜ ---
def main():
    print("--- ORİJİNAL RESNET18 TEST BAŞLIYOR ---")
    
    test_dataset = load_test_data(TEST_DATA_DIR)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Yeni (Saf) Mimarimizi yüklüyoruz
    model = OriginalResNet18ECG(NUM_CLASSES).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("✅ Şampiyon Model başarıyla yüklendi!")
    except Exception as e:
        print(f"❌ HATA: Model dosyası yüklenemedi! Hata: {e}")
        return

    model.eval()
    all_preds, all_targets = [], []
    
    MAX_SAMPLES = 20 
    counts = {'correct': 0, 'wrong': 0, 'suspicious': 0}
    
    print("Sinyaller analiz ediliyor, şüpheli vakalar aranıyor...")
    with torch.no_grad():
        batch_start = 0
        for inputs, targets in tqdm(test_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            mean = inputs.mean(dim=2, keepdim=True)
            std = inputs.std(dim=2, keepdim=True)
            inputs = (inputs - mean) / (std + 1e-5)
            
            outputs = model(inputs)
            probs_batch = F.softmax(outputs, dim=1) 
            _, preds = torch.max(probs_batch, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            current_batch_size = inputs.size(0)
            for i in range(current_batch_size):
                t_lbl = targets[i].item()
                p_lbl = preds[i].item()
                probs = probs_batch[i]
                
                top2_prob, _ = torch.topk(probs, 2)
                is_suspicious = (top2_prob[0].item() < 0.70) and (top2_prob[1].item() > 0.30)
                
                sig = inputs[i].cpu().numpy()
                sample_id = batch_start + i
                
                if is_suspicious and counts['suspicious'] < MAX_SAMPLES:
                    plot_ecg_sample(sig, t_lbl, probs, p_lbl, sample_id, os.path.join(RESULTS_DIR, "suspicious_samples"), is_suspicious=True)
                    counts['suspicious'] += 1
                elif t_lbl == p_lbl and counts['correct'] < MAX_SAMPLES and not is_suspicious:
                    plot_ecg_sample(sig, t_lbl, probs, p_lbl, sample_id, os.path.join(RESULTS_DIR, "correct_samples"))
                    counts['correct'] += 1
                elif t_lbl != p_lbl and counts['wrong'] < MAX_SAMPLES and not is_suspicious:
                    plot_ecg_sample(sig, t_lbl, probs, p_lbl, sample_id, os.path.join(RESULTS_DIR, "wrong_samples"))
                    counts['wrong'] += 1
                    
            batch_start += inputs.size(0)

    # 4. Raporlama
    print("\n--- RESNET18 TEST SONUÇLARI ---")
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro')
    report = classification_report(all_targets, all_preds, target_names=CLASS_NAMES)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print(report)
    
    with open(os.path.join(RESULTS_DIR, "resnet18_advanced_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Model: Orijinal Saf ResNet18 (Şampiyon)\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Macro F1: {f1:.4f}\n\n")
        f.write(report)

    plot_confusion_matrix(all_targets, all_preds, os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    print(f"\n✅ Tüm analizler, matris ve mor başlıklı şüpheli grafikler şuraya kaydedildi: {RESULTS_DIR}")

if __name__ == "__main__":
    main()