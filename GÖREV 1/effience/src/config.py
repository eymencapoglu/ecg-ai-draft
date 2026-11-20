
import os

# Base Paths
BASE_DIR = r"c:/Users/Eymen/Desktop/EKG/GÖREV 1"
DATASET1_DIR = os.path.join(BASE_DIR, "dataset 1", "a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0")
DATASET3_DIR = os.path.join(BASE_DIR, "dataset 3", "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")

# Hyperparameters
SAMPLE_RATE = 500
input_length = 5000  # 10 seconds at 500Hz
batch_size = 32
num_epochs = 10
learning_rate = 1e-3
num_classes = 3

# Classes
CLASS_NAMES = ["Normal", "Arrhythmia", "Block"]
# 0: Normal, 1: Arrhythmia, 2: Block

# SNOMED Mappings for Dataset 1
# Key: SNOMED Code (String), Value: Class Int
SNOMED_MAPPING = {
    # Normal / Sinus
    '426783006': 0, # SR (Sinus Rhythm)
    '426177001': 0, # SB (Sinus Bradycardia)
    '427084000': 0, # ST (Sinus Tachycardia)
    '427393009': 0, # SA (Sinus Irregularity)
    
    # Arrhythmias
    '164889003': 1, # AFIB
    '164890007': 1, # AF (Flutter)
    '251173003': 1, # ABI (Atrial Bigeminy) - classify as arrhythmia
    '284470004': 1, # APB (Atrial Premature Beats)
    '426761007': 1, # SVT
    '713422000': 1, # AT (Atrial Tach)
    '233896004': 1, # AVNRT
    '233897008': 1, # AVRT
    '11157007': 1,  # VB (Ventricular Bigeminy)
    '17338001': 1,  # VPB (Ventricular Premature Beat)
    '75532003': 1,  # VEB
    '426995002': 1, # JEB
    '251164006': 1, # JPT
    
    # Blocks (Conduction)
    '270492004': 2, # 1AVB
    '195042002': 2, # 2AVB
    '54016002': 2,  # 2AVB1
    '28189009': 2,  # 2AVB2
    '27885002': 2,  # 3AVB
    '233917008': 2, # AVB
    '164909002': 2, # LBBB (and variants encoded same)
    '59118001': 2,  # RBBB
    '698252002': 2, # IVB / IDC
    '164873001': 2, # LVH (Technically hypertrophy but often grouped with structural/conduction for specific tasks, but user said "Conduction Blocks". strict view: exclude LVH/RVH. I will exclude LVH from mapping to be safe, or ask. User said "Conduction Blocks", LVH is not a block. I will NOT map LVH to Block.)
    # Excluded: LVH, RVH, MI codes
}

# SCP Codes for Dataset 3 (PTB-XL)
# Key: SCP Code (String), Value: Class Int
SCP_MAPPING = {
    'NORM': 0,
    'SR': 0,
    
    'AFIB': 1,
    'AFLT': 1,
    'SVT': 1,
    'SVTAC': 1,
    'PSVT': 1,
    'BIGU': 1, 
    'TRIGU': 1,
    'PVC': 1,
    'PAC': 1,
    
    'LBBB': 2,
    'RBBB': 2,
    'CLBBB': 2,
    'CRBBB': 2,
    'ILBBB': 2,
    'IRBBB': 2,
    'AVB': 2,
    '1AVB': 2,
    '2AVB': 2,
    '3AVB': 2,
    'LAFB': 2,
    'LPFB': 2,
    'IVCD': 2,
    'WPW': 2, # Wolff-Parkinson-White, pre-excitation, conduction issue.
}
