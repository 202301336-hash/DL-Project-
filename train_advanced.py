import os
import sys
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 1. إعداد المسارات المباشرة للمشروع
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INNER_DIR = os.path.join(BASE_DIR, "ESC50_Final_Handoff")

for p in [BASE_DIR, INNER_DIR, os.path.join(INNER_DIR, "src"), os.path.join(INNER_DIR, "src", "models"), os.path.join(INNER_DIR, "data")]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from advanced_model import AdvancedSpectrogramNet

# استدعاء الـ Loader الحقيقي للبيانات
try:
    from dataset_loader import ESC50SpectrogramDataset
except ImportError:
    from src.dataset_loader import ESC50SpectrogramDataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def find_file(relative_path):
    p1 = os.path.join(INNER_DIR, relative_path)
    p2 = os.path.join(BASE_DIR, relative_path)
    if os.path.exists(p1): return p1
    if os.path.exists(p2): return p2
    return p1

def train_single_seed(seed, train_df, val_df, data_dir, epochs=20, batch_size=32, lr=0.0003):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "=" * 60)
    print(f"🚀 Training Seed {seed} on Device: {device}")
    print("=" * 60)

    # قراءة وتوليد الـ Log-Mel Spectrograms
    train_dataset = ESC50SpectrogramDataset(metadata_df=train_df, base_data_dir=data_dir)
    val_dataset = ESC50SpectrogramDataset(metadata_df=val_df, base_data_dir=data_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = AdvancedSpectrogramNet(num_classes=50).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001)

    save_dir = find_file("saved_models")
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"advanced_model_seed_{seed}.pth")

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # Validation Loop
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        train_acc = (correct / max(total, 1)) * 100
        val_acc = (val_correct / max(val_total, 1)) * 100
        elapsed = time.time() - start_time

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Seed {seed} | Epoch [{epoch:02d}/{epochs:02d}] - {elapsed:.1f}s | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    print(f"💾 Best Checkpoint saved to: {checkpoint_path} (Best Val Acc: {best_val_acc:.2f}%)")

def main():
    SEEDS = [42, 43, 44]
    
    # البحث عن ملف البيانات الوصفية (esc50.csv أو split_manifest)
    csv_path = find_file(os.path.join("results", "split_manifest_v2.csv"))
    if not os.path.exists(csv_path):
        csv_path = find_file(os.path.join("meta", "esc50.csv"))
    if not os.path.exists(csv_path):
        csv_path = find_file("esc50.csv")

    data_dir = find_file("audio")
    df = pd.read_csv(csv_path)

    # توحيد اسم عمود الـ target
    if 'target' not in df.columns and 'category_id' in df.columns:
        df['target'] = df['category_id']

    # تقسيم البيانات الموحد للفريق (Folds 1-4 للتدريب والـ Validation، و Fold 5 للـ Test المستقل)
    if 'split' in df.columns:
        train_df = df[df['split'].str.lower() == 'train'].reset_index(drop=True)
        val_df = df[df['split'].str.lower().isin(['val', 'validation'])].reset_index(drop=True)
    else:
        # التقسيم الموحد لـ 1600 عينة الأولى (Fold 1-4)
        non_test_df = df[df['fold'] != 5].reset_index(drop=True)
        train_df = non_test_df.iloc[:1280].reset_index(drop=True) # 1280 عينة
        val_df = non_test_df.iloc[1280:].reset_index(drop=True)  # 320 عينة

    print(f"--> Audio Dataset Loaded: {len(train_df)} Training Samples | {len(val_df)} Validation Samples")

    for seed in SEEDS:
        train_single_seed(seed, train_df, val_df, data_dir, epochs=20)

if __name__ == "__main__":
    main()