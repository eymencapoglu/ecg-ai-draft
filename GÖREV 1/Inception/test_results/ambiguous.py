import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from tqdm import tqdm
import warnings

# Gereksiz uyarıları kapat
warnings.filterwarnings("ignore")

# --- AYARLAR ---
DATA_SOURCE_DIR = r"C:\Users\Eymen\Desktop\EKG\GÖREV 1\düzeltilmiş_resnet_v2"
TEST_DATA_DIR = os.path.join(DATA_SOURCE_DIR, "benchmark")
MODEL_DIR = r"C:\Users\Eymen\Desktop\EKG\GÖREV 1\Inception"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
RESULTS_DIR = os.path.join(MODEL_DIR, "ambiguous_results")

# Klasörü oluştur
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1 # Tek tek analiz edeceğiz
CLASS_NAMES = ["Ritim (0)", "İletim (1)", "Normal (2)"]

# --- 1. MODEL MİMARİSİ (Aynı Olmalı) ---
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
    print(f"Test verileri taranıyor...")
    for f in files:
        try:
            data = torch.load(f)
            if torch.isnan(data['X']).any(): data['X'] = torch.nan_to_num(data['X'], nan=0.0)
            datasets.append(TensorDataset(data['X'], data['y']))
        except: pass
    return ConcatDataset(datasets)

def plot_ambiguous_sample(signal, true_lbl, probs, sample_id):
    """
    Sinyali ve modelin tüm sınıflara verdiği olasılıkları çizer.
    """
    leads = ['Lead I', 'Lead II', 'Lead III']
    t = np.arange(signal.shape[1])
    
    # En yüksek iki tahmin
    top2_prob, top2_idx = torch.topk(probs, 2)
    pred_lbl = top2_idx[0].item()
    confidence = top2_prob[0].item()
    
    # Başlık Rengi: Doğruysa Yeşil, Yanlışsa Kırmızı, Arada kaldıysa Sarı
    if true_lbl == pred_lbl:
        color = 'green' if confidence > 0.7 else 'orange' # Doğru ama güvensizse turuncu
    else:
        color = 'red'

    # Detaylı Başlık Metni
    title_text = f"Sample #{sample_id}\nGerçek: {CLASS_NAMES[true_lbl]} | Tahmin: {CLASS_NAMES[pred_lbl]}\n"
    title_text += f"Olasılıklar -> {CLASS_NAMES[0]}: %{probs[0]*100:.1f} | {CLASS_NAMES[1]}: %{probs[1]*100:.1f} | {CLASS_NAMES[2]}: %{probs[2]*100:.1f}"

    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    fig.suptitle(title_text, fontsize=14, color=color, weight='bold')

    for i in range(3):
        ax = axes[i]
        ax.plot(t, signal[i, :], label=leads[i], color='black', linewidth=0.8)
        ax.set_ylabel("mV")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc="upper right")
        
    axes[-1].set_xlabel("Zaman (Örnek)")
    
    # Dosya ismi: AMBIGUOUS_ID_Gercek_Tahmin.png
    filename = f"AMBIGUOUS_{sample_id}_True_{true_lbl}_Pred_{pred_lbl}.png"
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()

# --- 3. ANA ANALİZ DÖNGÜSÜ ---
def main():
    print("--- KARARSIZ / ZOR ÖRNEKLER ANALİZİ ---")
    
    # Veri ve Model Yükle
    test_dataset = load_test_data(TEST_DATA_DIR)
    # Hızlı analiz için 1000 örnek yeterli, hepsine bakmaya gerek yok
    subset_indices = np.random.choice(len(test_dataset), min(len(test_dataset), 2000), replace=False)
    test_loader = DataLoader(test_dataset, batch_size=1, sampler=torch.utils.data.SubsetRandomSampler(subset_indices))
    
    model = TeknofestAttentionModel(3).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print("Analiz yapılıyor, zorlu örnekler aranıyor...")
    
    found_count = 0
    MAX_SAMPLES_TO_SAVE = 20 # Çok fazla dosya üretmemek için limit
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(tqdm(test_loader)):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            # Normalizasyon
            mean = inputs.mean(dim=2, keepdim=True)
            std = inputs.std(dim=2, keepdim=True)
            inputs = (inputs - mean) / (std + 1e-5)
            
            # Model Çıktısı (Logits)
            outputs = model(inputs)
            
            # Softmax ile Olasılığa Çevir (0.0 - 1.0 arası)
            probs = F.softmax(outputs, dim=1)[0] # Tek örnek olduğu için [0]
            
            # En yüksek iki olasılığı al
            top2_val, top2_idx = torch.topk(probs, 2)
            
            top1_prob = top2_val[0].item()
            top2_prob = top2_val[1].item()
            
            # KRİTERLER:
            # 1. Fark Kriteri: Birinci ile İkinci arasındaki fark %15'ten azsa (Çok kararsız)
            diff_criterion = (top1_prob - top2_prob) < 0.15
            
            # 2. Güven Kriteri: En yüksek tahmin bile %60'ın altındaysa (Hiçbirinden emin değil)
            confidence_criterion = top1_prob < 0.60
            
            # 3. Hata Kriteri: Yanlış bildi ama ikinci tahmin doğruydu (Kıl payı kaçırdı)
            pred_lbl = top2_idx[0].item()
            true_lbl = targets.item()
            miss_criterion = (pred_lbl != true_lbl) and (top2_idx[1].item() == true_lbl) and ((top1_prob - top2_prob) < 0.25)

            if diff_criterion or confidence_criterion or miss_criterion:
                # Bu ilginç bir örnek! Kaydet.
                sig = inputs[0].cpu().numpy()
                plot_ambiguous_sample(sig, true_lbl, probs, i)
                found_count += 1
                
                if found_count >= MAX_SAMPLES_TO_SAVE:
                    print(f"\n{MAX_SAMPLES_TO_SAVE} adet kritik örnek bulundu ve kaydedildi.")
                    break
    
    if found_count == 0:
        print("\nModel çok emin! Hiç kararsız kaldığı örnek bulunamadı (Bu hem iyi hem kötü olabilir).")
    else:
        print(f"\nAnaliz Tamamlandı. Grafikler buraya kaydedildi: {RESULTS_DIR}")

if __name__ == "__main__":
    main()