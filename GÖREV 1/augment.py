import os
import wfdb
import torch
import numpy as np
import random
from tqdm import tqdm

# =====================================================================
# --- 1. YOLLAR VE AYARLAR ---
# =====================================================================
SOURCE_DIRS = [
    r"C:\Users\Eymen\Desktop\EKG\GÖREV 1\dataset 1",
    r"C:\Users\Eymen\Desktop\EKG\GÖREV 1\dataset 3",
    r"C:\Users\Eymen\Desktop\EKG\GÖREV 1\dataset 4"
]

TARGET_DIR = r"E:\augmented ekg data"
os.makedirs(TARGET_DIR, exist_ok=True)

CLASS_NAMES = {0: "Normal", 1: "İletim", 2: "Ritim"}

SNOMED_MAP = {
    '426783006': 0, '426177001': 0, '427084000': 0,
    '164909002': 1, '59118001': 1, '713427006': 1, '164905002': 1, '251146004': 1, '270492004': 1,
    '164889003': 2, '164890007': 2, '284470004': 2, '713426002': 2, '164884008': 2, '233917008': 2
}

def safe_augment_signal(sig):
    aug_type = random.choice(['noise', 'scale', 'shift'])
    if aug_type == 'noise':
        return sig + np.random.normal(0, 0.015, sig.shape)
    elif aug_type == 'scale':
        return sig * random.uniform(0.95, 1.05)
    elif aug_type == 'shift':
        return np.roll(sig, shift=random.randint(-50, 50), axis=1)
    return sig

def parse_snomed_from_hea(hea_path):
    try:
        with open(hea_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if line.startswith('#Dx:'):
                return line.strip().split(': ')[1].split(',')
    except:
        pass
    return []

def scan_and_register():
    print("🔍 1. AŞAMA: Tüm Kayıtlar Taranıyor (Sızıntılar Filtreleniyor)...")
    registry = {0: [], 1: [], 2: []}
    all_hea_files = []
    
    for d in SOURCE_DIRS:
        for root, dirs, files in os.walk(d):
            for file in files:
                if file.endswith(".hea"):
                    all_hea_files.append(os.path.join(root, file))

    for hea_file in tqdm(all_hea_files, desc="Tanılar Çözümleniyor"):
        if "_lr.hea" in hea_file: continue 
            
        snomed_codes = parse_snomed_from_hea(hea_file)
        assigned = list(set([SNOMED_MAP[c] for c in snomed_codes if c in SNOMED_MAP]))
        
        if len(assigned) == 1:
            registry[assigned[0]].append(hea_file.replace('.hea', ''))

    return registry

def process_and_augment(registry):
    print("\n⚖️ 2. AŞAMA: Orijinal Sayılar ve Akıllı Tavan (Smart Cap) Belirleniyor...")
    
    counts = {lbl: len(items) for lbl, items in registry.items()}
    for lbl, count in counts.items():
        print(f"   {CLASS_NAMES[lbl]} Sınıfı: {count} orijinal kayıt.")
        
    # --- AKILLI TAVAN (ORTANCA DEĞER) ---
    # Değerleri sırala ve ortancayı al
    sorted_counts = sorted(counts.values())
    target_count = sorted_counts[1] # En küçük=0. indeks, Ortanca=1. indeks, En büyük=2. indeks
    
    print(f"\n🎯 Hedeflenen Akıllı Denge: Her sınıf {target_count} veriye eşitlenecek!")
    print(f"   -> Normal Sınıfı {target_count} adede DÜŞÜRÜLECEK (Fazlalıklar atılacak)")
    print(f"   -> Ritim Sınıfı OLDUĞU GİBİ KALACAK")
    print(f"   -> İletim Sınıfı {target_count} adede ÇIKARILACAK (Güvenli 4x Artırma)\n")
    
    CHUNK_SIZE = 1500 
    current_X, current_y = [], []
    chunk_idx = 0
    
    for lbl, items in registry.items():
        print(f"İşleniyor: {CLASS_NAMES[lbl]} Sınıfı...")
        
        # Eğer veri sayısı hedeften büyükse KARIŞTIR ve KES (Undersampling)
        if len(items) > target_count:
            random.shuffle(items)
            items_to_process = items[:target_count]
        else:
            items_to_process = items
            
        valid_originals = []
        for record_path in tqdm(items_to_process, desc="Orijinaller Okunuyor"):
            try:
                sig, _ = wfdb.rdsamp(record_path)
                sig = sig.T
                if sig.shape[0] != 12 or np.isnan(sig).any() or np.isinf(sig).any(): continue
                
                if sig.shape[1] >= 5000: sig = sig[:, :5000]
                else: sig = np.pad(sig, ((0, 0), (0, 5000 - sig.shape[1])), 'constant')
                
                valid_originals.append(sig)
                current_X.append(torch.tensor(sig, dtype=torch.float32))
                current_y.append(lbl)
                
                if len(current_X) >= CHUNK_SIZE:
                    torch.save({'X': torch.stack(current_X), 'y': torch.tensor(current_y, dtype=torch.long)}, 
                               os.path.join(TARGET_DIR, f"aug_data_{CLASS_NAMES[lbl]}_part_{chunk_idx}.pt"))
                    current_X, current_y = [], []
                    chunk_idx += 1
            except: pass
                
        # Eğer veri sayısı hedeften küçükse ARTIR (Augmentation)
        if len(valid_originals) > 0 and len(valid_originals) < target_count:
            shortage = target_count - len(valid_originals)
            print(f"  -> {shortage} adet Augmented {CLASS_NAMES[lbl]} üretiliyor...")
            
            for _ in tqdm(range(shortage), desc="Augmentation"):
                base_sig = random.choice(valid_originals)
                aug_sig = safe_augment_signal(base_sig)
                
                current_X.append(torch.tensor(aug_sig, dtype=torch.float32))
                current_y.append(lbl)
                
                if len(current_X) >= CHUNK_SIZE:
                    torch.save({'X': torch.stack(current_X), 'y': torch.tensor(current_y, dtype=torch.long)}, 
                               os.path.join(TARGET_DIR, f"aug_data_{CLASS_NAMES[lbl]}_part_{chunk_idx}.pt"))
                    current_X, current_y = [], []
                    chunk_idx += 1

    if len(current_X) > 0:
        torch.save({'X': torch.stack(current_X), 'y': torch.tensor(current_y, dtype=torch.long)}, 
                   os.path.join(TARGET_DIR, f"aug_data_Final_part_{chunk_idx}.pt"))

    print("\n🎉 AKILLI DENGELEME OPERASYONU TAMAMLANDI!")
    print(f"Toplam {3 * target_count} adet kusursuz veri '{TARGET_DIR}' klasörüne taşındı.")

if __name__ == "__main__":
    reg = scan_and_register()
    process_and_augment(reg)