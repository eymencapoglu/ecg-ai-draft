import os
import torch
import numpy as np
import wfdb
from scipy import signal
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import gc

# --- AYARLAR ---
RAW_DATA_PATHS = [
    r"C:\Users\Eymen\Desktop\EKG\GÖREV 1\dataset 1",
    r"C:\Users\Eymen\Desktop\EKG\GÖREV 1\dataset 2",
    r"C:\Users\Eymen\Desktop\EKG\GÖREV 1\dataset 3",
    r"C:\Users\Eymen\Desktop\EKG\GÖREV 1\dataset 4"
]

# YENİ HEDEF KLASÖR
OUTPUT_DIR = r"C:\Users\Eymen\Desktop\EKG\GÖREV 1\düzeltilmiş_resnet_v2"
BENCHMARK_DIR = os.path.join(OUTPUT_DIR, "benchmark")
TRAIN_DIR = os.path.join(OUTPUT_DIR, "processed_train")

os.makedirs(BENCHMARK_DIR, exist_ok=True)
os.makedirs(TRAIN_DIR, exist_ok=True)

TARGET_LENGTH = 5000
FS = 500
BATCH_SIZE = 2000 

# --- GÜNCELLENMİŞ ETİKET MANTIĞI (Kritik Düzeltme) ---
def get_label(dx_list):
    dx_set = set(str(d) for d in dx_list)
    
    # Kapsamlı Kod Listeleri
    normal_codes = {'426783006', '164865005', 'NORMAL', 'SR', 'N'}
    conduction_codes = {'164909002', '59118001', '6374002', '270492004', '195042002', '713422000', '28189009', '445118002', '445211001', '698252002', 'LBBB', 'RBBB', 'AVB'}
    rhythm_codes = {'164889003', '164890007', '426177001', '427084000', '164934002', '59931005', '427393009', '426761007', '713427006', 'AF', 'AFIB', 'AFL', 'SVT'}

    # Öncelik Sıralaması:
    if not dx_set.isdisjoint(rhythm_codes): return 0 # Ritim
    if not dx_set.isdisjoint(conduction_codes): return 1 # İletim
    if not dx_set.isdisjoint(normal_codes): return 2 # Normal
    return -1

def apply_filter(data, fs):
    b, a = signal.butter(1, [0.5/(0.5*fs), 50.0/(0.5*fs)], btype='band')
    return signal.filtfilt(b, a, data, axis=0)

def parse_header(path, fields):
    codes = []
    if 'comments' in fields:
        for c in fields['comments']:
            if c.startswith('Dx:'): codes.extend(c.split(': ')[1].split(','))
    return codes

# --- İŞLEME DÖNGÜSÜ ---
all_records = []
print("Dosyalar taranıyor...")
for d in RAW_DATA_PATHS:
    for r, _, f in os.walk(d):
        for file in f:
            if file.endswith(".hea"): all_records.append(os.path.join(r, file[:-4]))

print(f"Toplam {len(all_records)} dosya bulundu. İşleniyor...")

batch_X, batch_y = [], []
part_counter = 1

for record_path in tqdm(all_records):
    try:
        signals, fields = wfdb.rdsamp(record_path)
        dx_list = parse_header(record_path, fields)
        y = get_label(dx_list)
        if y == -1: continue

        if fields['fs'] != FS:
            signals = signal.resample(signals, int(signals.shape[0] * FS / fields['fs']))
        signals = apply_filter(signals, FS)
        
        slen = signals.shape[0]
        if slen < TARGET_LENGTH:
            signals = np.vstack((signals, np.zeros((TARGET_LENGTH - slen, 12))))
        else:
            signals = signals[:TARGET_LENGTH, :]
            
        batch_X.append(signals.T.astype(np.float32))
        batch_y.append(y)

        if len(batch_X) >= BATCH_SIZE:
            X_arr, y_arr = np.array(batch_X), np.array(batch_y)
            # Stratify Split
            if len(np.unique(y_arr)) > 1:
                X_tr, X_temp, y_tr, y_temp = train_test_split(X_arr, y_arr, test_size=0.2, stratify=y_arr, random_state=42)
                X_val, X_te, y_val, y_te = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
                
                torch.save({'X': torch.tensor(X_tr), 'y': torch.tensor(y_tr)}, os.path.join(TRAIN_DIR, f"train_part_{part_counter}.pt"))
                torch.save({'X': torch.tensor(X_val), 'y': torch.tensor(y_val)}, os.path.join(BENCHMARK_DIR, f"val_part_{part_counter}.pt"))
                torch.save({'X': torch.tensor(X_te), 'y': torch.tensor(y_te)}, os.path.join(BENCHMARK_DIR, f"test_part_{part_counter}.pt"))
                
                part_counter += 1
                batch_X, batch_y = [], []
                gc.collect()
            else:
                batch_X, batch_y = [], [] # Tek tip sınıf gelirse batch'i atla (nadir durum)
    except: continue

print("Veri hazırlama tamamlandı.")