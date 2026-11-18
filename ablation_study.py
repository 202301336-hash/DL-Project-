import os
import sys
import time
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INNER_DIR = os.path.join(BASE_DIR, "ESC50_Final_Handoff")

for p in [BASE_DIR, INNER_DIR, os.path.join(INNER_DIR, "src"), os.path.join(INNER_DIR, "src", "models")]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from advanced_model import AdvancedSpectrogramNet, BaselineSpectrogramNet

def get_stats(model):
    params = sum(p.numel() for p in model.parameters())
    size_mb = (params * 4) / (1024 ** 2)
    return params, size_mb

def measure_inference_time(model, device, num_runs=100):
    model.eval()
    dummy_input = torch.randn(1, 1, 128, 431).to(device)
    for _ in range(10):
        _ = model(dummy_input)
        
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    total_time = time.time() - start_time
    return (total_time / num_runs) * 1000

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    baseline = BaselineSpectrogramNet(num_classes=50).to(device)
    cbam_model = AdvancedSpectrogramNet(num_classes=50).to(device)

    b_p, b_s = get_stats(baseline)
    c_p, c_s = get_stats(cbam_model)

    b_inf = measure_inference_time(baseline, device)
    c_inf = measure_inference_time(cbam_model, device)

    param_overhead = ((c_p - b_p) / b_p) * 100

    print("=" * 80)
    print("📊 MEMBER 4: ABLATION & COMPUTATIONAL COST ANALYSIS")
    print("=" * 80)
    print(f"{'Metric / Feature':<35} | {'Baseline (No CBAM)':<20} | {'Proposed (With CBAM)':<20}")
    print("-" * 80)
    print(f"{'Total Parameters':<35} | {b_p:<20,} | {c_p:<20,}")
    print(f"{'Model Memory Size':<35} | {b_s:.2f} MB{'':<13} | {c_s:.2f} MB")
    print(f"{'Parameter Overhead (%)':<35} | {'Reference (0.0%)':<20} | {f'+{param_overhead:.2f}%':<20}")
    print(f"{'Inference Time per Clip (ms)':<35} | {b_inf:.2f} ms{'':<13} | {c_inf:.2f} ms")
    print(f"{'Hardware Backend':<35} | {str(device).upper():<20} | {str(device).upper():<20}")
    print("=" * 80)

if __name__ == "__main__":
    main()