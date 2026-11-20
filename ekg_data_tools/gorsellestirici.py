import os
import glob
import random
import wfdb
import matplotlib.pyplot as plt
import numpy as np

def plot_12_leads(record_path, save_dir):
    """
    Tek bir EKG kaydının 12 derivasyonunu alt alta senkronize şekilde çizer.
    """
    try:
        record = wfdb.rdrecord(record_path)
        sig = record.p_signal
        fs = record.fs
        sig_names = record.sig_name
    except Exception as e:
        print(f"[HATA] {record_path} okunamadı. Detay: {e}")
        return

    # 12 Derivasyon için 12 satırlık ortak X eksenli figür oluştur
    fig, axs = plt.subplots(12, 1, figsize=(16, 22), sharex=True, dpi=200)
    record_id = os.path.basename(record_path)
    fig.suptitle(f"12 Derivasyonlu EKG Kıyaslama Tablosu: {record_id} (fs={fs}Hz)", fontsize=18, fontweight='bold', y=0.92)

    time_axis = np.arange(sig.shape[0]) / fs

    for i in range(min(12, sig.shape[1])):
        # Klinik EKG görünümü: Siyah dalga çizgileri
        axs[i].plot(time_axis, sig[:, i], linewidth=1.0, color='#111111')
        
        # Y ekseni başlıkları (Lead isimleri)
        axs[i].set_ylabel(sig_names[i], fontsize=12, fontweight='bold', rotation=0, labelpad=30, va='center')
        
        # Klinik EKG kağıdı görünümü için kırmızımtırak ızgara
        axs[i].grid(True, linestyle='-', alpha=0.3, color='#cc0000')
        axs[i].grid(True, which='minor', linestyle=':', alpha=0.2, color='#cc0000')
        axs[i].minorticks_on()
        
        # Y ekseni limitlerini standardize et (Aşırı sıçramaları filtrele)
        y_min, y_max = np.percentile(sig[:, i], [1, 99])
        axs[i].set_ylim(y_min - 0.5, y_max + 0.5)

        # X ekseni etiketini sadece en alt grafikte göster
        if i == 11:
            axs[i].set_xlabel("Zaman (Saniye)", fontsize=14, fontweight='bold')
        else:
            axs[i].tick_params(labelbottom=False)

    # Grafikleri birbirine sıkıştırarak zaman senkronizasyonunu netleştir
    plt.subplots_adjust(hspace=0.1)
    
    save_path = os.path.join(save_dir, f"{record_id}_12Lead_Comparison.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[BAŞARILI] Görsel oluşturuldu: {save_path}")

def generate_sample_plots(dataset_dir, num_samples=5):
    print(f"Taranıyor: {dataset_dir}")
    
    # Tüm .hea dosyalarını bul
    search_pattern = os.path.join(dataset_dir, "**", "*.hea")
    all_hea_files = glob.glob(search_pattern, recursive=True)
    
    if not all_hea_files:
        print("[HATA] Belirtilen klasörde .hea uzantılı dosya bulunamadı. Yolu kontrol edin.")
        return

    # Rastgele N adet kayıt seç
    selected_files = random.sample(all_hea_files, min(num_samples, len(all_hea_files)))
    
    output_dir = "EKG_Kiyaslama_Gorselleri"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Toplam {len(selected_files)} adet rastgele kayıt görselleştiriliyor...\n")
    
    for hea_file in selected_files:
        # .hea uzantısını atarak wfdb okuma formatına getir
        record_path = hea_file[:-4]
        plot_12_leads(record_path, output_dir)
        
    print(f"\nİşlem tamamlandı. Görseller '{os.path.abspath(output_dir)}' klasörüne kaydedildi.")

if __name__ == "__main__":
    # DİKKAT: Eski veri setinizin kök dizinini buraya girin.
    ESKI_VERI_SETI_YOLU = r"C:\Users\Eymen\Desktop\EKG\GÖREV 1\dataset 4" 
    
    # Kaç adet kayıt çizdirmek istiyorsanız num_samples değerini güncelleyin
    generate_sample_plots(ESKI_VERI_SETI_YOLU, num_samples=5)