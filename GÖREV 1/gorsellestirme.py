import numpy as np
import matplotlib.pyplot as plt
import wfdb
import random

def safe_augment_signal(sig):
    # 1. %1.5 Gauss Gürültüsü
    noise = np.random.normal(0, 0.015, sig.shape)
    sig_noisy = sig + noise
    
    # 2. Zaman Kaydırma
    shift_val = random.randint(20, 50)
    sig_shifted = np.roll(sig_noisy, shift=shift_val)
    
    return sig_shifted

def plot_ecg_pipeline(record_path, lead_index=1):
    print(f"Sinyal okunuyor: {record_path} ...")
    
    try:
        sig, meta = wfdb.rdsamp(record_path)
        raw_signal = sig[:, lead_index] 
    except Exception as e:
        print(f"HATA: Dosya okunamadı! Yolu kontrol et.\nDetay: {e}")
        return

    processed_signal = raw_signal.copy()
    
    # Boyut Standardizasyonu
    if len(processed_signal) >= 5000:
        processed_signal = processed_signal[:5000]
    else:
        pad_len = 5000 - len(processed_signal)
        processed_signal = np.pad(processed_signal, (0, pad_len), 'constant')
        
    # Z-Score Normalizasyonu
    mean_val = np.mean(processed_signal)
    std_val = np.std(processed_signal)
    if std_val > 0:
        processed_signal = (processed_signal - mean_val) / std_val

    augmented_signal = safe_augment_signal(processed_signal)

    # ==========================================
    # SADE VE DOĞRUDAN KAYIT İÇİN ÇİZİM AYARLARI
    # ==========================================
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), dpi=300) 

    # 1. GRAFİK: Ham Sinyal
    axs[0].plot(raw_signal, color='#2c3e50', linewidth=1)
    axs[0].set_title("1. Ham EKG Sinyali", fontsize=12, fontweight='bold')
    axs[0].set_ylabel("Voltaj (mV)", fontsize=10)
    axs[0].set_xlim(0, len(raw_signal))
    axs[0].grid(True, linestyle=':', alpha=0.5) # Daha hafif, noktalı ızgara

    # 2. GRAFİK: İşlenmiş Sinyal
    axs[1].plot(processed_signal, color='#2980b9', linewidth=1)
    axs[1].set_title("2. Saflaştırılmış Sinyal (5000 Örneklem, Z-Score)", fontsize=12, fontweight='bold')
    axs[1].set_ylabel("Normalize Genlik", fontsize=10)
    axs[1].set_xlim(0, 5000)
    axs[1].grid(True, linestyle=':', alpha=0.5)

    # 3. GRAFİK: Artırılmış Sinyal
    axs[2].plot(augmented_signal, color='#e67e22', linewidth=1)
    axs[2].set_title("3. Artırılmış Sinyal (Mikro Gürültü ve Zaman Kaydırma)", fontsize=12, fontweight='bold')
    axs[2].set_xlabel("Zaman Adımları (Örneklem Sayısı)", fontsize=10)
    axs[2].set_ylabel("Normalize Genlik", fontsize=10)
    axs[2].set_xlim(0, 5000)
    axs[2].grid(True, linestyle=':', alpha=0.5)

    # İç içe geçmeyi kesin olarak çözen otomatik düzenleyici
    plt.tight_layout(pad=2.0, h_pad=3.0)
    
    # Ekranda GÖSTERMEDEN doğrudan kaydet
    dosya_adi = "veri_onisleme_sade.png"
    plt.savefig(dosya_adi, format='png', bbox_inches='tight')
    
    # Ram'i temizle (arkaplanda açık kalmasın)
    plt.close(fig)
    
    print(f"✅ Çizim tamamlandı! Ekranda açılmadan doğrudan '{dosya_adi}' adıyla bulunduğun klasöre kaydedildi.")

if __name__ == "__main__":
    # Kendi yolunu buraya yapıştır
    DOSYA_YOLU = r"C:\Users\Eymen\Desktop\EKG\GÖREV 1\dataset 4\Training_WFDB\A0001" 
    plot_ecg_pipeline(DOSYA_YOLU)