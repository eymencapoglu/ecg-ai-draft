import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from torchvision.models import resnet101
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
from tqdm import tqdm
import warnings

# Gereksiz uyarıları kapat
warnings.filterwarnings("ignore")

# --- AYARLAR ---
# Verinin olduğu yer (Benchmark klasörü)
BASE_DIR = r"C:\Users\Eymen\Desktop\EKG\GÖREV 1\düzeltilmiş_resnet_v2"
TEST_DATA_DIR = os.path.join(BASE_DIR, "benchmark")

# Modelin kayıtlı olduğu klasör (Eğitim kodundaki MODEL_SAVE_DIR)
MODEL_DIR = os.path.join(BASE_DIR, "models") 
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")

# Sonuç Klasörü
RESULTS_DIR = os.path.join(MODEL_DIR, "test_results_resnet101")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "correct_samples"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "wrong_samples"), exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
NUM_CLASSES = 3
CLASS_NAMES = ["Ritim (0)", "İletim (1)", "Normal (2)"]

# --- 1. MODEL MİMARİSİ (ResNet101 + BiLSTM + Attention) ---

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
        if self.bias:
            self.param_bias.data.fill_(0)

    def forward(self, x):
        eij = torch.mm(x.contiguous().view(-1, self.feature_dim), self.param_weight).view(-1, self.step_dim)
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
        
        self.resnet = resnet101(weights=None)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.encoder = nn.Sequential(*list(self.resnet.children())[:-2])
        self.se_block = SEBlock(2048) 
        self.hidden_dim = 256
        self.lstm = nn.LSTM(
            input_size=2048, hidden_size=self.hidden_dim, num_layers=2, 
            batch_first=True, bidirectional=True, dropout=0.2
        )
        self.attention = Attention(feature_dim=self.hidden_dim * 2, step_dim=157) 
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, num_classes)
        )

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
    plt.title('Confusion Matrix (ResNet101 Hybrid)')
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
    print("--- RESNET101 HİBRİT MODEL TESTİ BAŞLIYOR ---")
    
    # 1. Veri Yükle
    test_dataset = load_test_data(TEST_DATA_DIR)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Toplam Test Verisi: {len(test_dataset)}")

    # 2. Modeli Yükle
    print(f"Model yükleniyor: {MODEL_PATH}")
    model = AdvancedECGModel(NUM_CLASSES).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"HATA: Model {MODEL_PATH} konumunda bulunamadı!")
        return

    model.eval()

    # 3. Tahmin
    all_preds = []
    all_targets = []
    
    MAX_SAMPLES = 20 
    collected_correct = 0
    collected_wrong = 0
    
    print("Tahminler yapılıyor...")
    
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
    
    with open(os.path.join(RESULTS_DIR, "final_test_report_resnet101.txt"), "w", encoding="utf-8") as f:
        f.write(f"Model: ResNet101 + BiLSTM + Attention\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Macro F1: {f1:.4f}\n\n")
        f.write(report)

    plot_confusion_matrix(all_targets, all_preds, os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    print(f"\nGrafikler ve Rapor kaydedildi: {RESULTS_DIR}")

if __name__ == "__main__":
    main()