# Environmental Sound Classification Under Noise (ESC-50)
**Deep Learning Final Project (A1403) — Fall 2026**  
**Faculty of Computer Science & Artificial Intelligence, Artificial Intelligence Department**  
**Pharos University in Alexandria (PUA)**  

* **Supervisor:** Dr. Sahar Ghanem  
* **Teaching Assistants:** Eng. Esraa Habiba, Eng. Philopater Hany, Eng. Mohamed Ashraf  
* **Chosen Project Idea:** #17 — Environmental Sound Classification Under Noise  
Team Name : TEAM OVERFITTED 
* **Team Members:**  
  1. Rania Hamdy Gomaa — 202302910 (Member 1 — Data & Baseline): ESC-50 EDA, Leakage-Free Splits, SNR Noise Mixing & Classical ML Baseline (SVM / Random Forest).

  2. Heba Atef Farouk — 202301959 (Member 2 — Classic Deep Learning): 1D-CNN, MLP & 2D-CNN Spectrogram Models, Hyperparameter & Noise Augmentation Studies 

  3. Roqaya Ebrahim Elghandour — 202302796 (Member 3 — Modern Deep Learning): Pretrained Audio Spectrogram Transformer (AST) Fine-Tuning & Pretrained-vs-Scratch Comparison.

  4. Nour Osama Hassan — 202301336 (Member 4 — Advanced Architecture & Ablation): Advanced Noise-Robust Architecture, Custom Component Implementation & Ablation Study 

  5. Shahd Emad — 202301173 (Member 5 — Evaluation & Live Demo): Shared Multi-Seed Evaluation Protocol, Error Analysis, Computational Cost & Gradio Live Demo 

---

## 1. Project Overview & Best Results
Environmental Sound Classification (ESC) in real-world environments suffers heavily from background noise degradation. In this project, we systematically evaluate classical machine learning baselines and three deep learning paradigms on the **ESC-50** dataset across clean audio and four additive white Gaussian noise levels (**10 dB, 5 dB, 0 dB, and -5 dB SNR**) under a strict **leakage-free fold split** (Folds 1–3 Train, Fold 4 Validation, Fold 5 Test) averaged across **3 random seeds (42, 123, 2026)**.

### Key Experimental Results (Fold 5 Test Set — Mean Accuracy %)
| Model Family | Architecture | Clean | 10 dB | 5 dB | 0 dB | -5 dB |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | Classical ML (SVM / Random Forest) | ~48.50% | ~28.00% | ~19.50% | ~11.25% | ~6.50% |
| **Model 1 (Tabular/1D)** | MLP / 1D-CNN on Acoustic Features | ~54.25% | ~35.00% | ~24.75% | ~15.00% | ~8.75% |
| **Model 2 (Classic DL)** | Custom 2D-CNN on Log-Mel Spectrograms | ~68.50% | ~52.25% | ~41.00% | ~22.50% | ~12.00% |
| **Model 3 (Modern/Pretrained)** | **Audio Spectrogram Transformer (AST)** | **95.83%** | **93.17%** | **90.17%** | **85.75%** | **76.25%** |

* **Best Overall Performance:** Fine-tuned **Audio Spectrogram Transformer (AST)** achieved **95.83% ± 0.14%** test accuracy on clean audio and maintained **76.25% ± 0.66%** accuracy under extreme **-5 dB** noise (outperforming 2D-CNN by +64.25% at -5 dB).

---

## 2. Repository Structure
```text
ESC50_Final_Handoff/
├── README.md                        # Project documentation and instructions
├── requirements.txt                 # Unified Python dependencies
├── project_config.py                # Global paths, seeds, and SNR configurations
├── src/                             # Leakage-free data loading, SNR mixing, and feature extraction
├── scripts/                         # Preprocessing and baseline helper scripts
├── run_all_models.py                # Baseline & Model 1 (SVM, RF, MLP, 1D-CNN) training
├── train_advanced.py                # Advanced 1D/feature model training
├── evaluate_advanced.py             # Cross-SNR evaluation script
├── ablation_study.py                # Ablation experiments (noise augmentation & feature removal)
├── Member 2_classic_dl/             # Model 2: Custom 2D-CNN implementation & checkpoints
├── Member 3_ast/                    # Model 3: Audio Spectrogram Transformer (AST) probe & experiments
├── Final_Team_Delivery/             # Aggregated comparison tables (JSON/CSV) across all models
├── demo/
│   └── app.py                       # Interactive Gradio Blind Live Demo (with real-time SNR slider & mic)
├── docs/                            # Technical report, presentation slides, and contribution logs
└── meta/                            # ESC-50 metadata (esc50.csv)
```

---

## 3. Environment Setup
1. 1. **Clone the repository:**
   ```bash
   git clone [https://github.com/202301336-hash/DL-Project-.git](https://github.com/202301336-hash/DL-Project-.git)
   cd DL-Project- 
   ```
2. **Create and activate a virtual environment (Python 3.10+ recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```
3. **Install all required packages:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 4. Dataset & Pretrained Checkpoints (Google Drive)
To keep the repository lightweight and compliant with submission guidelines, raw `.wav` audio files (`audio/`) and large model checkpoints (`saved_models/`, `best_head.pth`) are hosted externally on Google Drive:
* - **Google Drive Link (Dataset + Trained Checkpoints):** [https://drive.google.com/drive/folders/1IAYF4KTnJqK1bQm7YAHCmo7WuzYPi0Xr?usp=sharing](https://drive.google.com/drive/folders/1IAYF4KTnJqK1bQm7YAHCmo7WuzYPi0Xr?usp=sharing) 

### Dataset Setup Instructions:
1. Download `ESC-50` from the Google Drive link above (or the official repository: `https://github.com/karolpiczak/ESC-50`).
2. Extract the `audio/` folder (2,000 `.wav` clips, 5 seconds each, 50 balanced classes) and `meta/esc50.csv` into the root directory of this project.
3. Place the downloaded checkpoint `best_head.pth` into `Member 3_ast/experiments/ast_probe/42/best_head.pth`.

---

## 5. Training Commands
All models use the exact same fold split (`Folds 1-3` Train, `Fold 4` Val, `Fold 5` Test) to prevent source-file data leakage:

* **Baseline & Model 1 (SVM, Random Forest, MLP, 1D-CNN):**
  ```bash
  python run_all_models.py
  python train_advanced.py
  ```
* **Model 2 (Custom 2D-CNN on Log-Mel Spectrograms):**
  ```bash
  python "Member 2_classic_dl/train_cnn2d.py"
  ```
* **Model 3 (Audio Spectrogram Transformer — AST Fine-Tuning across 3 Seeds):**
  ```bash
  python "Member 3_ast/train_ast.py" --seeds 42 123 2026
  ```

---

## 6. Evaluation & Ablation Commands
* **Evaluate all models across Clean, 10 dB, 5 dB, 0 dB, and -5 dB SNR levels:**
  ```bash
  python evaluate_advanced.py
  python generate_final_comparison.py
  ```
* **Run Ablation Study (impact of noise-aware augmentation and pretraining):**
  ```bash
  python ablation_study.py
  ```
* Consolidated test metrics are saved automatically inside `Final_Team_Delivery/`.

---

## 7. Running the Live Demo
Launch the interactive Gradio application (`demo/app.py`), which supports **unlabeled Fold 5 test samples**, **live microphone recording**, and a **real-time SNR noise slider (-5 dB to 40 dB)**:
```bash
python demo/app.py
```
Once started, open the local URL printed in the terminal (typically `http://127.0.0.1:7860`).

---

## 8. Hardware Used & Expected Runtimes
* **Hardware:** NVIDIA GPU (CUDA-enabled, e.g., RTX 3060 / T4 16GB VRAM) & Intel Core i7 CPU (16GB RAM).
* **Expected Runtimes:**
  * Feature Extraction & Classical Baseline (SVM/RF): ~2 minutes (CPU)
  * Model 2 (2D-CNN, 3 seeds): ~12 minutes (GPU)
  * Model 3 (AST Fine-Tuning, 3 seeds): ~18 minutes (GPU)
  * Live Demo Inference: < 0.15 seconds per 5-second audio clip

---

## 9. Credits & References
1. **Dataset:** Piczak, K. J. (2015). *ESC: Dataset for Environmental Sound Classification*. Proceedings of the 23rd ACM International Conference on Multimedia.
2. **Pretrained Backbone:** Gong, Y., Chung, Y. A., & Glass, J. (2021). *AST: Audio Spectrogram Transformer*. Interspeech 2021 (`MIT/ast-finetuned-audioset-10-10-0.4593` via Hugging Face Transformers).
3. **Custom Components:** Power-calibrated SNR white noise mixer (`mix_noise`), leakage-free source-grouped split auditor, and blind Gradio inference pipeline developed by TEAM OVERFITTED.