import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
from scipy.io import wavfile
import torchaudio.transforms as T
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split 

# ==========================================
# 1. إعداد المسارات وحفظ المخرجات
# ==========================================
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results/logs", exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "../audio"          # مسار مجلد الصوتيات في المشروع الرئيسي
META_CSV = "../meta/esc50.csv" # مسار ملف البيانات في المشروع الرئيسي

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# ==========================================
# 2. تجهيز البيانات + Noise Augmentation
# ==========================================
class ESC50AudioDataset(Dataset):
    def __init__(self, df, audio_dir, sample_rate=22050, duration=5, 
                 aug_noise_snr=None, eval_noise_snr=None, spec_aug=False):
        self.df = df.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.target_len = sample_rate * duration
        self.aug_noise_snr = aug_noise_snr
        self.eval_noise_snr = eval_noise_snr
        self.spec_aug = spec_aug

        self.mel_spec = T.MelSpectrogram(
            sample_rate=sample_rate, n_fft=1024, hop_length=512, n_mels=128
        )
        self.amplitude_to_db = T.AmplitudeToDB()
        self.freq_mask = T.FrequencyMasking(freq_mask_param=15)
        self.time_mask = T.TimeMasking(time_mask_param=35)

    def add_awgn_noise(self, waveform, snr_db):
        sig_power = waveform.pow(2).mean()
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = sig_power / snr_linear
        noise = torch.randn_like(waveform) * torch.sqrt(noise_power + 1e-10)
        return waveform + noise

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = os.path.join(self.audio_dir, row['filename'])
        sr, audio_np = wavfile.read(filepath)
        waveform = torch.from_numpy(audio_np.astype(np.float32) / 32768.0).unsqueeze(0)
        if waveform.ndim == 3:
           waveform = waveform.squeeze(0).T
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if waveform.shape[1] < self.target_len:
            waveform = nn.functional.pad(waveform, (0, self.target_len - waveform.shape[1]))
        else:
            waveform = waveform[:, :self.target_len]

        # إضافة ضوضاء عشوائية أثناء التدريب
        if self.aug_noise_snr is not None and random.random() < 0.5:
            chosen_snr = random.choice(self.aug_noise_snr)
            waveform = self.add_awgn_noise(waveform, chosen_snr)

        # إضافة ضوضاء ثابتة أثناء التقييم
        if self.eval_noise_snr is not None:
            waveform = self.add_awgn_noise(waveform, self.eval_noise_snr)

        spec = self.amplitude_to_db(self.mel_spec(waveform))
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)

        if self.spec_aug:
            spec = self.freq_mask(spec)
            spec = self.time_mask(spec)

        return spec, int(row['target'])

def get_splits():
    manifest_path = "../data/splits/split_manifest_v2.csv"
    if os.path.exists(manifest_path):
        df = pd.read_csv(manifest_path)
        train_df = df[df['split'] == 'train'].copy()
        val_df = df[df['split'] == 'val'].copy()
        test_df = df[df['split'] == 'test'].copy()
        return train_df, val_df, test_df
    else:
        df = pd.read_csv(META_CSV)
        test_df = df[df['fold'] == 5].copy()
        train_val_df = df[df['fold'] != 5].copy()
        train_df, val_df = train_test_split(
            train_val_df, test_size=320, stratify=train_val_df['target'], random_state=42
        )
        return train_df, val_df, test_df

# ==========================================
# 3. المعمارية المختارة: 2D-CNN
# ==========================================
class Audio2DCNN(nn.Module):
    def __init__(self, num_classes=50, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout * 0.5),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout * 0.5),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ==========================================
# 4. التدريب والتقييم وحفظ الـ Logs و Checkpoints
# ==========================================
def train_and_eval(lr=1e-3, dropout=0.3, use_noise_aug=True, seed=42, epochs=40, exp_name="baseline"):
    print(f"\n--- Running Experiment: {exp_name} (Seed: {seed}) ---")
    set_seed(seed)
    train_df, val_df, test_df = get_splits()
    
    aug_snrs = [20, 15, 10, 5] if use_noise_aug else None
    train_ds = ESC50AudioDataset(train_df, DATA_DIR, aug_noise_snr=aug_snrs, spec_aug=True)
    val_ds = ESC50AudioDataset(val_df, DATA_DIR)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    model = Audio2DCNN(num_classes=50, dropout=dropout).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0.0
    history = []
    ckpt_path = f"checkpoints/{exp_name}_seed{seed}.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for x, y in val_loader:
                out = model(x.to(DEVICE))
                preds.extend(out.argmax(dim=1).cpu().numpy())
                targets.extend(y.numpy())
        
        val_acc = accuracy_score(targets, preds)
        history.append({"epoch": epoch, "train_loss": round(train_loss/len(train_loader), 4), "val_acc": round(val_acc*100, 2)})
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)

    pd.DataFrame(history).to_csv(f"results/logs/{exp_name}_seed{seed}_log.csv", index=False)
    
    # تقييم الموديل الأفضل عبر مستويات الضوضاء المختلفة على Test Set (400 عينة)
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    
    noise_levels = [None, 20, 10, 5, 0]
    robustness_results = {}
    
    for snr in noise_levels:
        test_ds = ESC50AudioDataset(test_df, DATA_DIR, eval_noise_snr=snr)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
        preds, targets = [], []
        with torch.no_grad():
            for x, y in test_loader:
                out = model(x.to(DEVICE))
                preds.extend(out.argmax(dim=1).cpu().numpy())
                targets.extend(y.numpy())
        label = "clean" if snr is None else f"snr_{snr}dB"
        robustness_results[label] = {
            "accuracy": round(accuracy_score(targets, preds) * 100, 2),
            "macro_f1": round(f1_score(targets, preds, average="macro") * 100, 2)
        }
        
    print(f"Best Val Acc: {best_val_acc*100:.2f}% | Clean Test Acc: {robustness_results['clean']['accuracy']}%")
    return round(best_val_acc * 100, 2), robustness_results

# ==========================================
# 5. تشغيل جميع الدراسات المطلوبة
# ==========================================
if __name__ == "__main__":
    summary_report = {}

    # 1. Hyperparameter Study
    hp_configs = [
        {"name": "hp_lr1e-3_drop0.3", "lr": 1e-3, "dropout": 0.3},
        {"name": "hp_lr5e-4_drop0.4", "lr": 5e-4, "dropout": 0.4},
    ]
    for cfg in hp_configs:
        val_acc, res = train_and_eval(lr=cfg["lr"], dropout=cfg["dropout"], use_noise_aug=True, seed=42, exp_name=cfg["name"])
        summary_report[cfg["name"]] = {"val_acc": val_acc, "test_robustness": res}

    # 2. Noise-Augmentation Study (بدون Noise Augmentation)
    val_no_aug, res_no_aug = train_and_eval(lr=1e-3, dropout=0.3, use_noise_aug=False, seed=42, exp_name="no_noise_aug")
    summary_report["no_noise_aug"] = {"val_acc": val_no_aug, "test_robustness": res_no_aug}

    # 3. Repeated Random Seeds (42, 43, 44)
    seeds = [42, 43, 44]
    seed_results = []
    for s in seeds:
        _, res = train_and_eval(lr=1e-3, dropout=0.3, use_noise_aug=True, seed=s, exp_name="final_2dcnn")
        seed_results.append(res)
    
    summary_report["repeated_seeds_summary"] = {}
    for level in ["clean", "snr_20dB", "snr_10dB", "snr_5dB", "snr_0dB"]:
        accs = [r[level]["accuracy"] for r in seed_results]
        summary_report["repeated_seeds_summary"][level] = f"{np.mean(accs):.2f}% ± {np.std(accs):.2f}%"

    with open("results/final_experiment_summary.json", "w") as f:
        json.dump(summary_report, f, indent=4)
    print("\n✅ تم الانتهاء من جميع التجارب وحفظ النتائج بنجاح!")