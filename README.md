# 12-Lead ECG Multi-Class Classification — Research & Prototyping Archive

*[English](#english) | [Türkçe](#türkçe)*

---

<a name="english"></a>
## English

### Overview
This repository archives our early-stage research and iterative prototyping for **12-Lead ECG Multi-Class Classification** across three clinical categories mapped via **SNOMED-CT** diagnostic codes:
- **Normal**
- **Conduction Disorders (*İletim Bozuklukları*)**
- **Rhythm Disorders / Arrhythmias (*Ritim Bozuklukları*)**

Over several months of experimentation (originally developed as our draft pipeline for TEKNOFEST Health AI), we tackled class imbalance, patient/data leakage prevention, multi-dataset merging, signal augmentation, and systematic benchmarking across CNN, Attention, Transformer/Conformer, and Ensemble architectures.

### Development Timeline & Iterative Progression
Based on original file timestamps, the project evolved through clear research milestones:
1. **Feb 3–4, 2026 — Baseline CNNs (`GÖREV 1/effience/`, `GÖREV 1/resnet101/`):** First 12-lead classification baselines with EfficientNet-B0 and ResNet-101. Early experiments exposed severe class-imbalance collapse (69% accuracy, 0.27 Macro F1 on ResNet-101), motivating better preprocessing and balancing.
2. **Feb 6–8, 2026 — Refined Preprocessing & Hybrid ResNet (`GÖREV 1/düzeltilmiş resnet101/`):** Built dedicated preprocessing (`preprocessing.py`, `preprocess_v2.py`), signal augmentation (`augment.py`, `augment_v2.py`), and `ResNet-18` / `ResNet-101` hybrid experiments.
3. **Feb 16–21, 2026 — Multi-Scale Inception & Dual-Class Analysis (`GÖREV 1/Inception/`, `düzeltilmiş_resnet_v2/`):** Introduced a multi-scale Attention-Inception architecture (`Teknofest Attention`) reaching **88.10% accuracy / 0.7809 Macro F1** (n=3,839), alongside ambiguous & dual-class diagnostic inspection.
4. **Mar 6–11, 2026 — Custom 1D Conformer, Optuna & Master Ensemble (`GÖREV 1/custom_transformer/`, `GÖREV 1/ENSEMBLE/`, `data_pure.py`):** Designed a custom 1D Conformer (`CONFORMER COLAB.ipynb`, `augmented_optuna.py`, `finetune_optuna.py` — **86.22% accuracy / 0.7707 Macro F1**), built `data_pure.py` to separate pure single-label vs. multi-label SNOMED-CT records without leakage, and combined top models in `ENSEMBLE/final_ensemble_test.py` reaching **93% accuracy / 0.86 Macro F1** (n=3,837).
5. **Mar 23–27, 2026 — Preprocessing Visuals & Interactive Report (`gorsellestirme.py`, `index.html`):** Generated signal-cleaning comparison figures and an interactive HTML summary report.
6. **May 9, 2026 — 12-Lead Clinical Waveform Visualizer (`ekg_data_tools/gorsellestirici.py`, `EKG_Kiyaslama_Gorselleri/`):** Built a clinical ECG-paper-style 12-lead synchronized plotter using `wfdb`.

### Benchmark Summary (From Saved Evaluation Reports)
| Model / Stage | Test Accuracy | Macro F1 | Support | Report |
| :--- | :---: | :---: | :---: | :--- |
| ResNet-101 (Initial Baseline) | 69.0% | 0.27 | 5,000 | `GÖREV 1/resnet101/outputs/classification_report.txt` |
| Custom Conformer1D | 86.22% | 0.7707 | 3,839 | `GÖREV 1/custom_transformer/test_results/conformer_test_report.txt` |
| Teknofest Attention (Inception) | 88.10% | 0.7809 | 3,839 | `GÖREV 1/Inception/test_results/final_test_report.txt` |
| **Master Ensemble** | **93.0%** | **0.86** | **3,837** | `GÖREV 1/ENSEMBLE/ensemble_report.txt` |

### Note on Excluded Data & Weights
To make this repository suitable for GitHub, ~22.6 GB of raw/processed ECG waveform datasets (`dataset 1–4`, `processed_train`, `benchmark`, `birleşik dataset`), `.pth` model checkpoints, and bulk per-sample prediction PNGs were excluded. All training/evaluation scripts, notebooks, reports, confusion matrices, and 12-lead comparison figures are preserved.

### Acknowledgments & Open-Source Credits
This research would not have been possible without the open-access biomedical datasets and tools provided by the scientific community. We gratefully acknowledge:
- **[PTB-XL ECG Dataset (v1.0.3)](https://physionet.org/content/ptb-xl/1.0.3/)** — Wagner et al., *Scientific Data* (2020), hosted on PhysioNet.
- **[A Large Scale 12-Lead Electrocardiogram Database for Arrhythmia Study (v1.0.0)](https://physionet.org/content/ecg-arrhythmia/1.0.0/)** — Zheng et al. (Chapman University & Shaoxing People's Hospital / Ningbo First Hospital), hosted on PhysioNet.
- **[PhysioNet / Computing in Cardiology Challenge 2020 (`Training_WFDB` / CPSC 2018)](https://physionet.org/content/challenge-2020/1.0.2/)** — Perez Alday et al.
- **[WFDB Python Package](https://github.com/MIT-LCP/wfdb-python)** (MIT Laboratory for Computational Physiology), **[PyTorch](https://pytorch.org/)**, **[Optuna](https://optuna.org/)**, and **[SNOMED-CT](https://www.snomed.org/)** clinical terminology.

---

<a name="türkçe"></a>
## Türkçe

### Genel Bakış
Bu repo, **12 Derivasyonlu EKG Sinyallerinin Çok Sınıflı Sınıflandırılması** üzerine gerçekleştirdiğimiz ilk aşama araştırma, veri ön işleme ve model kıyaslama çalışmalarının arşividir. **SNOMED-CT** tanı kodları kullanılarak kayıtlar üç ana klinik sınıfa ayrılmıştır:
- **Normal**
- **İletim Bozuklukları**
- **Ritim Bozuklukları (Aritmi)**

TEKNOFEST Sağlıkta Yapay Zeka hazırlık sürecinde geliştirilen bu taslak çalışmada; sınıf dengesizliği, veri sızıntısının (data leakage) önlenmesi, çoklu açık kaynak veri setlerinin birleştirilmesi, sinyal artırma (augmentation) ve farklı derin öğrenme mimarilerinin (CNN, Attention-Inception, 1D Conformer, Ensemble) karşılaştırılması adım adım ele alınmıştır.

### Geliştirme Süreci ve Kronoloji
Orijinal dosya tarihlerine göre projenin ilerleyişi:
1. **3–4 Şubat 2026 — İlk CNN Denemeleri (`effience/`, `resnet101/`):** EfficientNet-B0 ve ResNet-101 ile temel modeller kuruldu. İlk ResNet-101 denemesinde sınıf dengesizliği nedeniyle modelin çoğunluk sınıfına çöktüğü (%69 doğruluk, 0.27 Macro F1) görülerek veri ön işleme ve dengeleme stratejisine geçildi.
2. **6–8 Şubat 2026 — Geliştirilmiş Ön İşleme ve Hibrit ResNet (`düzeltilmiş resnet101/`):** Sinyal temizleme (`preprocessing.py`, `preprocess_v2.py`), veri artırma (`augment.py`, `augment_v2.py`) ve hibrit ResNet-18/101 eğitimleri yapıldı.
3. **16–21 Şubat 2026 — Çok Ölçekli Inception ve Çift Tanı Analizi (`Inception/`, `düzeltilmiş_resnet_v2/`):** `Teknofest Attention (Inception)` mimarisi ile **%88.10 doğruluk ve 0.7809 Macro F1** skoruna ulaşıldı; belirsiz (ambiguous) ve çok tanılı vakalar analiz edildi.
4. **6–11 Mart 2026 — Özel 1D Conformer, Optuna ve Master Ensemble (`custom_transformer/`, `ENSEMBLE/`, `data_pure.py`):** Özel 1D Conformer mimarisi (`CONFORMER COLAB.ipynb`, Optuna hiperparametre araması — **%86.22 doğruluk / 0.7707 Macro F1**) ve tek/çok tanılı kayıtları sızıntısız ayıran `data_pure.py` geliştirildi. En iyi modellerin birleştirildiği `Master Ensemble` ile **%93 doğruluk ve 0.86 Macro F1** elde edildi.
5. **23–27 Mart 2026 — Görselleştirme ve Raporlama (`gorsellestirme.py`, `index.html`):** Veri ön işleme görselleri ve etkileşimli HTML sonuç raporu oluşturuldu.
6. **9 Mayıs 2026 — 12 Derivasyonlu Klinik EKG Çizici (`ekg_data_tools/gorsellestirici.py`, `EKG_Kiyaslama_Gorselleri/`):** `wfdb` tabanlı klinik EKG kağıdı görünümünde 12 derivasyonlu kıyaslama görselleri üretildi.

### Hariç Tutulan Büyük Dosyalar
GitHub uyumluluğu için ~22.6 GB boyutundaki ham/işlenmiş EKG veri setleri (`dataset 1–4`, `processed_train`, `benchmark`), model ağırlıkları (`.pth`) ve yüzlerce tekil örnek tahmini hariç tutulmuş; tüm kaynak kodlar, notebook'lar, değerlendirme raporları, karmaşıklık matrisleri ve 12 derivasyonlu kıyaslama grafikleri korunmuştur.

### Açık Kaynak Teşekkürler ve Kaynaklar
Bu çalışmada yararlandığımız açık erişimli veri setlerine ve kütüphanelere teşekkür ederiz:
- **[PTB-XL ECG Dataset (v1.0.3)](https://physionet.org/content/ptb-xl/1.0.3/)** — Wagner vd. (*Scientific Data*, 2020), PhysioNet.
- **[A Large Scale 12-Lead Electrocardiogram Database for Arrhythmia Study (v1.0.0)](https://physionet.org/content/ecg-arrhythmia/1.0.0/)** — Zheng vd. (Chapman-Shaoxing / Ningbo), PhysioNet.
- **[PhysioNet / Computing in Cardiology Challenge 2020 (`Training_WFDB` / CPSC 2018)](https://physionet.org/content/challenge-2020/1.0.2/)** — Perez Alday vd.
- **[WFDB Python](https://github.com/MIT-LCP/wfdb-python)**, **[PyTorch](https://pytorch.org/)**, **[Optuna](https://optuna.org/)** ve **[SNOMED-CT](https://www.snomed.org/)**.