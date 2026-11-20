import os
import glob
import torch
import wfdb
import numpy as np
import random
from tqdm import tqdm

# =====================================================================
# --- 1. AYARLAR VE YOLLAR ---
# =====================================================================
SOURCE_DIRS = [
    r"C:\Users\Eymen\Desktop\EKG\GÖREV 1\dataset 1",
    r"C:\Users\Eymen\Desktop\EKG\GÖREV 1\dataset 3",
    r"C:\Users\Eymen\Desktop\EKG\GÖREV 1\dataset 4"
]

# Ana Klasör ve Alt Klasörler
BASE_TARGET_DIR = r"E:\ekg data"
PURE_DIR = os.path.join(BASE_TARGET_DIR, "pure_single_class")
MULTI_DIR = os.path.join(BASE_TARGET_DIR, "complex_multi_class")

os.makedirs(PURE_DIR, exist_ok=True)
os.makedirs(MULTI_DIR, exist_ok=True)

CLASS_NAMES = {0: "Normal", 1: "İletim", 2: "Ritim"}

# SNOMED-CT Haritası
SNOMED_MAP = {
    # Normal
    '426783006': 0, '426177001': 0, '427084000': 0,
    # İletim Bozuklukları
    '164909002': 1, '59118001': 1, '713427006': 1, '164905002': 1, '251146004': 1, '270492004': 1,
    # Ritim Bozuklukları
    '164889003': 2, '164890007': 2, '284470004': 2, '713426002': 2, '164884008': 2, '233917008': 2
}

# =====================================================================
# --- 2. VERİ AYRIŞTIRMA (TEK TANILI vs ÇİFT TANILI) ---
# =====================================================================
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

def scan_and_separate_data():
    print("🔍 1. AŞAMA: Ham Veriler Taranıyor ve Ayrıştırılıyor...")
    
    pure_registry = {0: [], 1: [], 2: []}
    multi_registry = []
    leak_prevented_count = 0
    
    all_hea_files = []
    for d in SOURCE_DIRS:
        for root, dirs, files in os.walk(d):
            for file in files:
                if file.endswith(".hea"):
                    all_hea_files.append(os.path.join(root, file))

    for hea_file in tqdm(all_hea_files, desc="Tanılar Sınıflandırılıyor"):
        # PTB-XL Düşük Çözünürlük (Sızıntı) Engeli
        if "_lr.hea" in hea_file:
            leak_prevented_count += 1
            continue
            
        snomed_codes = parse_snomed_from_hea(hea_file)
        assigned_classes = list(set([SNOMED_MAP[code] for code in snomed_codes if code in SNOMED_MAP]))
        
        record_name = hea_file.replace('.hea', '')
        
        if len(assigned_classes) == 1:
            # SAF KAN (TEK TANILI)
            pure_registry[assigned_classes[0]].append(record_name)
        elif len(assigned_classes) > 1:
            # KARMAŞIK VAKA (ÇİFT/ÇOK TANILI)
            multi_registry.append({'record': record_name, 'labels': assigned_classes})

    print(f"\n🛡️ Ayrıştırma Raporu:")
    print(f"  Saf (Tek Tanılı) Veri Adayları -> Normal: {len(pure_registry[0])}, İletim: {len(pure_registry[1])}, Ritim: {len(pure_registry[2])}")
    print(f"  Karmaşık (Çift Tanılı) Gelecek Rezerve Veriler -> {len(multi_registry)} adet")
    print(f"  Sızıntı Önleme Nedeniyle Atlananlar: {leak_prevented_count}")
    
    return pure_registry, multi_registry

# =====================================================================
# --- 3. SİNYAL OKUMA VE DİSKE YAZMA MOTORU ---
# =====================================================================
def process_and_save_signals(record_list, target_folder, is_multi_label=False, chunk_size=1000, desc="İşleniyor"):
    current_X, current_y = [], []
    chunk_idx = 0
    corrupted_signals = 0
    
    for i, item in enumerate(tqdm(record_list, desc=desc)):
        if is_multi_label:
            record_path = item['record']
            # Multi-hot encoding (Örn: Hem İletim Hem Ritim -> [0, 1, 1])
            labels = item['labels']
            y_val = torch.zeros(3, dtype=torch.float32)
            y_val[labels] = 1.0
        else:
            record_path = item['record']
            y_val = item['label']
            
        try:
            sig, meta = wfdb.rdsamp(record_path)
            sig = sig.T # [12, Length]
            
            if sig.shape[0] != 12 or np.isnan(sig).any() or np.isinf(sig).any():
                corrupted_signals += 1
                continue
                
            # Standardizasyon (5000)
            if sig.shape[1] >= 5000:
                sig = sig[:, :5000]
            else:
                pad_len = 5000 - sig.shape[1]
                sig = np.pad(sig, ((0, 0), (0, pad_len)), 'constant')
                
            current_X.append(torch.tensor(sig, dtype=torch.float32))
            current_y.append(y_val)
            
        except Exception:
            corrupted_signals += 1
            
        # Diske Yaz
        if len(current_X) >= chunk_size or i == len(record_list) - 1:
            if len(current_X) > 0:
                X_tensor = torch.stack(current_X)
                if is_multi_label:
                    y_tensor = torch.stack(current_y) # [Batch, 3]
                else:
                    y_tensor = torch.tensor(current_y, dtype=torch.long) # [Batch]
                    
                save_path = os.path.join(target_folder, f"data_part_{chunk_idx}.pt")
                torch.save({'X': X_tensor, 'y': y_tensor}, save_path)
                
                current_X, current_y = [], []
                chunk_idx += 1
                
    return chunk_idx, corrupted_signals

def main():
    pure_reg, multi_reg = scan_and_separate_data()
    
    # 1. Saf Verileri Dengele
    print("\n⚖️ 2. AŞAMA: Saf Veriler İçin Denge Kuruluyor...")
    min_count = min([len(items) for items in pure_reg.values()])
    print(f"🎯 Hedeflenen Saf Denge: Her sınıf için tam {min_count} veri.")
    
    balanced_pure_records = []
    for lbl, items in pure_reg.items():
        random.shuffle(items)
        for rec in items[:min_count]:
            balanced_pure_records.append({'record': rec, 'label': lbl})
    random.shuffle(balanced_pure_records)
    
    # 2. Saf Verileri İşle ve Kaydet
    print(f"\n🚀 3. AŞAMA: Saf Veriler '{PURE_DIR}' Konumuna Paketleniyor...")
    chunks_pure, corrupt_pure = process_and_save_signals(
        balanced_pure_records, PURE_DIR, is_multi_label=False, desc="Saf Veriler Yazılıyor"
    )
    
    # 3. Çift Tanılı Verileri İşle ve Kaydet (Dengeleme yapılmaz, tamamı alınır)
    print(f"\n🚀 4. AŞAMA: Çift Tanılı Karmaşık Veriler '{MULTI_DIR}' Konumuna Paketleniyor...")
    chunks_multi, corrupt_multi = process_and_save_signals(
        multi_reg, MULTI_DIR, is_multi_label=True, desc="Karmaşık Veriler Yazılıyor"
    )
    
    print("\n🎉 BÜYÜK RESET VE AYRIŞTIRMA TAMAMLANDI!")
    print(f"✅ Safkan Veri Paketleri: {chunks_pure} adet")
    print(f"✅ Çift Tanılı (Multi-label) Veri Paketleri: {chunks_multi} adet")
    print(f"⚠️ Okuma Hatası Veren Toplam Bozuk Sinyal: {corrupt_pure + corrupt_multi}")

if __name__ == "__main__":
    main()