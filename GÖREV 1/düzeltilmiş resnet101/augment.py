import torch
import os
import glob
import numpy as np
from tqdm import tqdm

# --- AYARLAR ---
BASE_DIR = r"C:\Users\Eymen\Desktop\EKG\GÖREV 1\düzeltilmiş_resnet_v2"
TRAIN_DIR = os.path.join(BASE_DIR, "processed_train")

# Hedef: Sınıf 1 (İletim) ve Sınıf 2 (Normal) sayılarını Sınıf 0'a yaklaştırmak
# İletim (4k) -> x5 kat artırılacak -> ~20k
# Normal (7k) -> x3 kat artırılacak -> ~21k
AUGMENT_MULTIPLIERS = {
    0: 0,  # Ritim: Zaten çok fazla, artırma
    1: 5,  # İletim: 5 kopya üret
    2: 3   # Normal: 3 kopya üret
}

def add_noise(signal):
    """Sinyale rastgele gürültü ekler"""
    noise = torch.randn_like(signal) * 0.05
    return signal + noise

def time_shift(signal):
    """Sinyali zamanda sağa/sola kaydırır"""
    shift = np.random.randint(-200, 200) # +/- 200 örnek kaydır
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
        signal = X_batch[i] # (12, 5000)
        
        # Eğer bu sınıf için artırma isteniyorsa
        multiplier = AUGMENT_MULTIPLIERS.get(label, 0)
        
        if multiplier > 0:
            for _ in range(multiplier):
                # Rastgele bir teknik seç
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
files = glob.glob(os.path.join(TRAIN_DIR, "train_part_*.pt"))
# Daha önce oluşturulmuş augmented dosyaları varsa atla (tekrar tekrar şişmesin)
files = [f for f in files if "aug" not in f]

print(f"{len(files)} adet eğitim dosyası tarandı. Azınlık sınıfları çoğaltılıyor...")

total_new_samples = 0

for f in tqdm(files):
    try:
        data = torch.load(f)
        X_orig, y_orig = data['X'], data['y']
        
        # Sadece artırılacak sınıfların olduğu verileri gönder
        aug_X, aug_y = apply_augmentation(X_orig, y_orig)
        
        if aug_X is not None:
            # Yeni dosya adı: train_part_1_aug.pt
            new_filename = f.replace(".pt", "_aug.pt")
            torch.save({'X': aug_X, 'y': aug_y}, new_filename)
            total_new_samples += len(aug_y)
            
    except Exception as e:
        print(f"Hata ({f}): {e}")

print(f"\nİŞLEM TAMAMLANDI!")
print(f"Toplam {total_new_samples} adet YENİ yapay veri üretildi ve kaydedildi.")
print("Eğitim kodunu tekrar çalıştırdığında bu dosyalar otomatik dahil edilecek.")