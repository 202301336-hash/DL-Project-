import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INNER_DIR = os.path.join(BASE_DIR, "ESC50_Final_Handoff")

for p in [BASE_DIR, INNER_DIR, os.path.join(INNER_DIR, "src"), os.path.join(INNER_DIR, "src", "models"), os.path.join(INNER_DIR, "data")]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from advanced_model import AdvancedSpectrogramNet

try:
    from dataset_loader import ESC50SpectrogramDataset
except ImportError:
    from src.dataset_loader import ESC50SpectrogramDataset

def find_file(relative_path):
    p1 = os.path.join(INNER_DIR, relative_path)
    p2 = os.path.join(BASE_DIR, relative_path)
    if os.path.exists(p1): return p1
    if os.path.exists(p2): return p2
    return p1

def add_gaussian_noise(waveforms, snr_db):
    if snr_db is None:
        return waveforms
    sig_power = torch.mean(waveforms ** 2, dim=(-2, -1), keepdim=True)
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    noise = torch.randn_like(waveforms) * torch.sqrt(noise_power)
    return waveforms + noise

def evaluate_condition(model, test_loader, device, snr_db=None):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = add_gaussian_noise(inputs, snr_db).to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    acc = np.mean(np.array(all_preds) == np.array(all_targets)) * 100
    _, _, f1_macro, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)
    return acc, f1_macro * 100

def main():
    SEEDS = [42, 43, 44]
    NOISE_LEVELS = [None, 10, 5, 0, -5]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    split_path = find_file(os.path.join("results", "split_manifest_v2.csv"))
    meta_path = find_file(os.path.join("meta", "esc50.csv"))
    data_dir = find_file("audio")

    if os.path.exists(split_path):
        df = pd.read_csv(split_path)
        if 'target' not in df.columns and 'category_id' in df.columns:
            df['target'] = df['category_id']
        test_df = df[df['split'] == 'test'].reset_index(drop=True)
    else:
        df = pd.read_csv(meta_path)
        if 'target' not in df.columns and 'category_id' in df.columns:
            df['target'] = df['category_id']
        test_df = df[df['fold'] == 5].reset_index(drop=True)

    test_dataset = ESC50SpectrogramDataset(metadata_df=test_df, base_data_dir=data_dir)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print("=" * 70)
    print("📊 ADVANCED MODEL EVALUATION (3 SEEDS & NOISE LEVELS)")
    print("=" * 70)

    results = {snr: {'acc': [], 'f1': []} for snr in NOISE_LEVELS}
    save_dir = find_file("saved_models")

    for seed in SEEDS:
        model = AdvancedSpectrogramNet(num_classes=50).to(device)
        model_path = os.path.join(save_dir, f"advanced_model_seed_{seed}.pth")
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print(f"--> Warning: Model checkpoint for seed {seed} not found! Skipping...")
            continue

        for snr in NOISE_LEVELS:
            acc, f1 = evaluate_condition(model, test_loader, device, snr)
            results[snr]['acc'].append(acc)
            results[snr]['f1'].append(f1)

    print("\n FINAL TEST RESULTS (Mean ± SD over 3 Seeds):")
    print("-" * 65)
    print(f"{'Condition':<15} | {'Accuracy (%)':<22} | {'Macro F1 (%)':<22}")
    print("-" * 65)

    for snr in NOISE_LEVELS:
        cond_name = "Clean" if snr is None else f"{snr} dB"
        accs = results[snr]['acc'] if len(results[snr]['acc']) > 0 else [0]
        f1s = results[snr]['f1'] if len(results[snr]['f1']) > 0 else [0]
        print(f"{cond_name:<15} | {np.mean(accs):.2f} ± {np.std(accs):.2f}%            | {np.mean(f1s):.2f} ± {np.std(f1s):.2f}%")
    print("=" * 65)

if __name__ == "__main__":
    main()