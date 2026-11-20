import torch
import os
import glob
import numpy as np
from tqdm import tqdm

# --- AYARLAR ---
# DİKKAT: V2 klasörünü hedefliyoruz
BASE_DIR = r"C:\Users\Eymen\Desktop\EKG\GÖREV 1\düzeltilmiş_resnet_v2"
TRAIN_DIR = os.path.join(BASE_DIR, "processed_train")

AUGMENT_MULTIPLIERS = {
    0: 0,  # Ritim: Artırma yok
    1: 5,  # İletim: 5 kat artır
    2: 3   # Normal: 3 kat artır
}

def add_noise(signal):
    return signal + torch.randn_like(signal) * 0.05

def time_shift(signal):
    shift = np.random.randint(-200, 200)
    return torch.roll(signal, shifts=shift, dims=1)

def scale_signal(signal):
    return signal * np.random.uniform(0.9, 1.1)

def apply_augmentation(X_batch, y_batch):
    new_X, new_y = [], []
    for i in range(len(y_batch)):
        label = y_batch[i].item()
        multiplier = AUGMENT_MULTIPLIERS.get(label, 0)
        
        if multiplier > 0:
            signal = X_batch[i]
            for _ in range(multiplier):
                aug_type = np.random.choice(['noise', 'shift', 'scale'])
                if aug_type == 'noise': aug_sig = add_noise(signal)
                elif aug_type == 'shift': aug_sig = time_shift(signal)
                else: aug_sig = scale_signal(signal)
                
                new_X.append(aug_sig)
                new_y.append(label)
                
    if len(new_X) > 0: return torch.stack(new_X), torch.tensor(new_y)
    return None, None

files = glob.glob(os.path.join(TRAIN_DIR, "train_part_*.pt"))
files = [f for f in files if "aug" not in f] # Zaten augmented olanları atla

print(f"{len(files)} dosya üzerinde veri arttırma yapılıyor...")

for f in tqdm(files):
    try:
        data = torch.load(f)
        aug_X, aug_y = apply_augmentation(data['X'], data['y'])
        
        if aug_X is not None:
            new_filename = f.replace(".pt", "_aug.pt")
            torch.save({'X': aug_X, 'y': aug_y}, new_filename)
    except: pass

print("Veri arttırma tamamlandı! Eğitim kodlarını çalıştırabilirsin.")