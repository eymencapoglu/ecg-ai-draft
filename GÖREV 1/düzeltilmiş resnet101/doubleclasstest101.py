import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from torchvision.models import resnet101
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# --- AYARLAR ---
BASE_DIR = r"C:\Users\Eymen\Desktop\EKG\GÖREV 1\düzeltilmiş_resnet_v2"
TEST_DATA_DIR = os.path.join(BASE_DIR, "benchmark")

# Modelin kayıtlı olduğu klasör
MODEL_DIR = os.path.join(BASE_DIR, "models") 
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")

# Sonuç Klasörü
RESULTS_DIR = os.path.join(MODEL_DIR, "dual_class_samples")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1 # Tek tek analiz edeceğiz
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
        if self.bias: self.param_bias.data.fill_(0)
    def forward(self, x):
        eij = torch.mm(x.contiguous().view(-1, self.feature_dim), self.param_weight).view(-1, self.step_dim)
        if self.bias: eij = eij + self.param_bias
        eij = torch.tanh(eij)
        a = self.soft_max(eij)
        weighted_input = x * a.unsqueeze(2).expand_as(x)
        return torch.sum(weighted_input, 1)

class AdvancedECGModel(nn.Module):
    def __init__(self, num_classes):
        super(AdvancedECGModel, self).__init__()
        self.resnet = resnet101(weights=None)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.encoder = nn.Sequential(*list(self.resnet.children())[:-2])
        self.se_block = SEBlock(2048)
        self.hidden_dim = 256
        self.lstm = nn.LSTM(input_size=2048, hidden_size=self.hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.attention = Attention(feature_dim=self.hidden_dim * 2, step_dim=157)
        self.fc = nn.Sequential(nn.Linear(self.hidden_dim * 2, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_classes))

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

# --- 2. VERİ YÜKLEME VE GÖRSELLEŞTİRME ---
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

def plot_dual_class_sample(signal, true_lbl, probs, sample_id):
    leads = ['Lead I', 'Lead II', 'Lead III']
    t = np.arange(signal.shape[1])
    
    top2_prob, top2_idx = torch.topk(probs, 2)
    class_1 = CLASS_NAMES[top2_idx[0].item()]
    class_2 = CLASS_NAMES[top2_idx[1].item()]
    prob_1 = top2_prob[0].item() * 100
    prob_2 = top2_prob[1].item() * 100
    
    title_text = f"ÇİFT TANILI SİNYAL (Sample #{sample_id})\n"
    title_text += f"Ana Etiket: {CLASS_NAMES[true_lbl]}\n"
    title_text += f"Modelin Gördükleri -> 1. {class_1}: %{prob_1:.1f} | 2. {class_2}: %{prob_2:.1f}"

    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    fig.suptitle(title_text, fontsize=14, color='purple', weight='bold')

    for i in range(3):
        ax = axes[i]
        ax.plot(t, signal[i, :], label=leads[i], color='black', linewidth=0.8)
        ax.set_ylabel("mV")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc="upper right")
        
    axes[-1].set_xlabel("Zaman (Örnek)")
    
    filename = f"DUAL_{sample_id}_T_{true_lbl}_P1_{top2_idx[0].item()}_P2_{top2_idx[1].item()}.png"
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()

# --- 3. ANA DÖNGÜ ---
def main():
    print("--- AYNI ANDA 2 SINIF İÇEREN (ÇİFT TANILI) ÖRNEK ANALİZİ ---")
    
    test_dataset = load_test_data(TEST_DATA_DIR)
    # Veri setini karıştırarak tarayalım ki farklı örnekler gelsin
    subset_indices = np.random.choice(len(test_dataset), min(len(test_dataset), 3000), replace=False)
    test_loader = DataLoader(test_dataset, batch_size=1, sampler=torch.utils.data.SubsetRandomSampler(subset_indices))
    
    model = AdvancedECGModel(3).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    found_count = 0
    MAX_SAMPLES_TO_SAVE = 20 # 20 tane bulana kadar devam et
    
    print("Model sinyallerde 'Çift Hastalık' (Dual Class) arıyor...")
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(tqdm(test_loader)):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            mean = inputs.mean(dim=2, keepdim=True)
            std = inputs.std(dim=2, keepdim=True)
            inputs = (inputs - mean) / (std + 1e-5)
            
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)[0]
            
            top2_val, top2_idx = torch.topk(probs, 2)
            
            # KRİTER: Model 1. tahmine de 2. tahmine de %35'in üzerinde olasılık veriyorsa
            # Bu demektir ki model "Bu sinyalde kesinlikle 2 farklı durum var" diyor.
            if top2_val[0].item() > 0.35 and top2_val[1].item() > 0.35:
                sig = inputs[0].cpu().numpy()
                true_lbl = targets.item()
                plot_dual_class_sample(sig, true_lbl, probs, i)
                found_count += 1
                
                if found_count >= MAX_SAMPLES_TO_SAVE:
                    print(f"\n{MAX_SAMPLES_TO_SAVE} adet 'Çift Tanılı' (İki Sınıflı) örnek bulundu!")
                    break
    
    if found_count == 0:
        print("\nModel hiç çift tanılı durum bulamadı (Sinyaller çok net ayrışmış olabilir).")
    else:
        print(f"\nAnaliz Tamamlandı. Grafikler (Mor Başlıklı) şuraya kaydedildi: {RESULTS_DIR}")

if __name__ == "__main__":
    main()