import torch
import os
import glob
import numpy as np
from tqdm import tqdm

# --- AYARLAR ---
BASE_DIR = r"C:\Users\Eymen\Desktop\EKG\GÖREV 1\düzeltilmiş_resnet_v2"
TRAIN_DIR = os.path.join(BASE_DIR, "processed_train")

# HEDEF: İletim (Class 1) Recall skorunu uçurmak için x10 yapıyoruz.
AUGMENT_MULTIPLIERS = {
    0: 0,   # Ritim: Dokunma
    1: 10,  # İletim: 10 katına çıkar (Agresif)
    2: 4    # Normal: 4 katına çıkar
}

def add_noise(signal):
    """Sinyale rastgele gürültü ekler"""
    noise = torch.randn_like(signal) * 0.05
    return signal + noise

def time_shift(signal):
    """Sinyali zamanda sağa/sola kaydırır"""
    shift = np.random.randint(-200, 200)
    return torch.roll(signal, shifts=shift, dims=1)

def scale_signal(signal):
    """Sinyal genliğini hafifçe değiştirir"""
    factor = np.random.uniform(0.9, 1.1)
    return signal * factor

def apply_augmentation(X_batch, y_batch):
    new_X = []
    new_y = []
    
    for i in range(len(y_batch)):
        label = y_batch[i].item()
        signal = X_batch[i]
        
        multiplier = AUGMENT_MULTIPLIERS.get(label, 0)
        
        if multiplier > 0:
            for _ in range(multiplier):
                aug_type = np.random.choice(['noise', 'shift', 'scale'])
                
                if aug_type == 'noise':
                    aug_sig = add_noise(signal)
                elif aug_type == 'shift':
                    aug_sig = time_shift(signal)
                else:
                    aug_sig = scale_signal(signal)
                
                new_X.append(aug_sig)
                new_y.append(label)
                
    if len(new_X) > 0:
        return torch.stack(new_X), torch.tensor(new_y)
    else:
        return None, None

# --- İŞLEM ---
print(f"Hedef Klasör: {TRAIN_DIR}")
files = glob.glob(os.path.join(TRAIN_DIR, "train_part_*.pt"))
# Zaten artırılmış dosyaları tekrar artırma!
files = [f for f in files if "aug" not in f]

print(f"{len(files)} adet ham eğitim dosyası bulundu. Hibrit model için agresif artırma başlıyor...")

total_new = 0
for f in tqdm(files):
    try:
        data = torch.load(f)
        aug_X, aug_y = apply_augmentation(data['X'], data['y'])
        
        if aug_X is not None:
            new_filename = f.replace(".pt", "_aug_hybrid.pt")
            torch.save({'X': aug_X, 'y': aug_y}, new_filename)
            total_new += len(aug_y)
    except Exception as e:
        print(f"Hata ({f}): {e}")

print(f"\nİŞLEM TAMAMLANDI! Toplam {total_new} yeni veri üretildi.")