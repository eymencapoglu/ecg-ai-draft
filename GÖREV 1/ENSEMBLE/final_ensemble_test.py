import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from torchvision.models import resnet18, resnet101
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm

# =====================================================================
# --- 1. YOLLAR VE YAPILANDIRMA ---
# =====================================================================
BASE_DIR = r"C:\Users\Eymen\Desktop\EKG\GÖREV 1"
BENCHMARK_DIR = os.path.join(BASE_DIR, "düzeltilmiş_resnet_v2", "benchmark")
OUTPUT_DIR = os.path.join(BASE_DIR, "ENSEMBLE")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['Normal', 'İletim', 'Ritim']

MODEL_PATHS = {
    "Conformer": os.path.join(BASE_DIR, "custom_transformer", "BEST_FINETUNED_AUGMENTED_CONFORMER_COLAB.pth"),
    "Res18_Hybrid": os.path.join(BASE_DIR, "düzeltilmiş_resnet_v2", "models_hybrid_final", "best_model.pth"),
    "Res18_Std": os.path.join(BASE_DIR, "düzeltilmiş_resnet_v2", "models_resnet18_std", "best_model.pth"),
    "Res101_Hybrid": os.path.join(BASE_DIR, "düzeltilmiş_resnet_v2", "models", "best_model.pth")
}

# =====================================================================
# --- 2. MİMARİ TANIMLARI (BOYUTLAR SABİTLENDİ) ---
# =====================================================================

class Conformer_Classifier(nn.Module):
    def __init__(self, in_chans=12, embed_dim=128, depth=4, heads=4, dropout=0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_chans, embed_dim // 2, kernel_size=15, stride=5, padding=7),
            nn.BatchNorm1d(embed_dim // 2), nn.GELU(),
            nn.Conv1d(embed_dim // 2, embed_dim, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(embed_dim), nn.GELU()
        )
        self.num_patches = 250
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=heads, dim_feedforward=embed_dim*4, dropout=dropout, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.fc = nn.Sequential(nn.Linear(embed_dim, embed_dim // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(embed_dim // 2, 3))
    def forward(self, x):
        features = self.stem(x)
        features = features.permute(0, 2, 1) + self.pos_embed
        encoded = self.encoder(features)
        return self.fc(torch.mean(encoded, dim=1))

class OriginalResNet18ECG(nn.Module):
    def __init__(self, num_classes=3):
        super(OriginalResNet18ECG, self).__init__()
        self.resnet = resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    def forward(self, x): return self.resnet(x.unsqueeze(1))

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False), nn.ReLU(inplace=True), nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim):
        super(Attention, self).__init__()
        self.feature_dim, self.step_dim = feature_dim, step_dim
        self.param_weight = nn.Parameter(torch.Tensor(feature_dim, 1))
        self.param_bias = nn.Parameter(torch.Tensor(1))
        nn.init.xavier_uniform_(self.param_weight)
        self.param_bias.data.fill_(0)
    def forward(self, x):
        eij = torch.tanh(torch.mm(x.contiguous().view(-1, self.feature_dim), self.param_weight).view(-1, self.step_dim) + self.param_bias)
        a = F.softmax(eij, dim=1)
        return torch.sum(x * a.unsqueeze(2).expand_as(x), 1)

class AdvancedECGModel(nn.Module):
    def __init__(self, num_classes, base_model="resnet101", hidden_dim=256, intermediate_dim=128):
        super(AdvancedECGModel, self).__init__()
        if base_model == "resnet101":
            self.resnet = resnet101(weights=None)
            in_ch = 2048
        else:
            self.resnet = resnet18(weights=None)
            in_ch = 512
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.encoder = nn.Sequential(*list(self.resnet.children())[:-2])
        self.se_block = SEBlock(in_ch)
        self.lstm = nn.LSTM(input_size=in_ch, hidden_size=hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.attention = Attention(feature_dim=hidden_dim * 2, step_dim=157)
        self.fc = nn.Sequential(nn.Linear(hidden_dim * 2, intermediate_dim), nn.ReLU(), nn.Dropout(0.3), nn.Linear(intermediate_dim, num_classes))
    def forward(self, x):
        x = self.encoder(x.unsqueeze(1))
        x = self.se_block(x).squeeze(2).permute(0, 2, 1)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        return self.fc(self.attention(x))

# =====================================================================
# --- 3. ENSEMBLE TEST VE KAYIT SÜRECİ ---
# =====================================================================

def run_ensemble():
    print("🧠 Modeller hafızaya yükleniyor...")
    models = {}
    models["Conformer"] = Conformer_Classifier().to(DEVICE)
    models["Conformer"].load_state_dict(torch.load(MODEL_PATHS["Conformer"], map_location=DEVICE))
    
    models["Res18_Std"] = OriginalResNet18ECG().to(DEVICE)
    models["Res18_Std"].load_state_dict(torch.load(MODEL_PATHS["Res18_Std"], map_location=DEVICE))
    
    models["Res18_Hybrid"] = AdvancedECGModel(num_classes=3, base_model="resnet18", hidden_dim=128, intermediate_dim=64).to(DEVICE)
    models["Res18_Hybrid"].load_state_dict(torch.load(MODEL_PATHS["Res18_Hybrid"], map_location=DEVICE))
    
    models["Res101_Hybrid"] = AdvancedECGModel(num_classes=3, base_model="resnet101", hidden_dim=256, intermediate_dim=128).to(DEVICE)
    models["Res101_Hybrid"].load_state_dict(torch.load(MODEL_PATHS["Res101_Hybrid"], map_location=DEVICE))

    for m in models.values(): m.eval()

    files = glob.glob(os.path.join(BENCHMARK_DIR, "val_part*.pt"))
    test_ds = ConcatDataset([TensorDataset(torch.load(f)['X'], torch.load(f)['y']) for f in files])
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    results_list = []
    print(f"🔬 {len(test_ds)} örnek üzerinde Master Ensemble analizi başladı...")

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(tqdm(test_loader)):
            inputs = inputs.to(DEVICE)
            mean, std = inputs.mean(dim=2, keepdim=True), inputs.std(dim=2, keepdim=True)
            norm_inputs = (inputs - mean) / (std + 1e-5)
            
            # Olasılıkları Topla
            p_conf = F.softmax(models["Conformer"](norm_inputs), dim=1).cpu().numpy()[0]
            p_r18s = F.softmax(models["Res18_Std"](norm_inputs), dim=1).cpu().numpy()[0]
            p_r18h = F.softmax(models["Res18_Hybrid"](norm_inputs), dim=1).cpu().numpy()[0]
            p_r101h = F.softmax(models["Res101_Hybrid"](norm_inputs), dim=1).cpu().numpy()[0]
            
            # Ağırlıklı Ortalama
            final_probs = (p_conf * 0.35) + (p_r18s * 0.20) + (p_r18h * 0.20) + (p_r101h * 0.25)
            final_pred = np.argmax(final_probs)
            target = targets.item()
            
            results_list.append({
                "target": target,
                "pred": final_pred,
                "prob_0": final_probs[0], "prob_1": final_probs[1], "prob_2": final_probs[2]
            })

    # --- KAYIT VE ANALİZ BÖLÜMÜ ---
    df = pd.DataFrame(results_list)
    df.to_csv(os.path.join(OUTPUT_DIR, "final_ensemble_results.csv"), index=False)
    
    report = classification_report(df['target'], df['pred'], target_names=CLASS_NAMES)
    with open(os.path.join(OUTPUT_DIR, "ensemble_report.txt"), "w") as f:
        f.write("MASTER ENSEMBLE NIHAI SONUCLAR\n" + "="*30 + "\n")
        f.write(report)
    
    print("\n🏆 MASTER ENSEMBLE SONUÇLARI:\n", report)

    # Karmaşıklık Matrisi Kaydetme
    cm = confusion_matrix(df['target'], df['pred'])
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='magma', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Master Ensemble (4 Model) Karmaşıklık Matrisi")
    plt.ylabel('Gerçek Etiket')
    plt.xlabel('Tahmin Edilen Etiket')
    plt.savefig(os.path.join(OUTPUT_DIR, "final_confusion_matrix.png"), dpi=300)
    plt.close()
    
    print(f"✅ Tüm analizler '{OUTPUT_DIR}' klasörüne kaydedildi.")

if __name__ == "__main__":
    run_ensemble()