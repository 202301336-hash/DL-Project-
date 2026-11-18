import os
import torch
import torchaudio
import numpy as np
import pandas as pd
import gradio as gr
from scipy.io import wavfile
from transformers import ASTFeatureExtractor, ASTForAudioClassification

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if not os.path.exists(os.path.join(BASE_DIR, "meta")):
    BASE_DIR = os.path.abspath(".")

# تحميل ملف الأوزان المدرب (best_head.pth)
ckpt_path = os.path.join(BASE_DIR, "Member 3_ast", "experiments", "ast_probe", "42", "best_head.pth")
ckpt = torch.load(ckpt_path, map_location=DEVICE)

MODEL_ID = ckpt.get("model_id", "MIT/ast-finetuned-audioset-10-10-0.4593")
LABELS = ckpt.get("labels", None)

# تحميل بيانات ESC-50
meta_path = os.path.join(BASE_DIR, "meta", "esc50.csv")
df_meta = pd.read_csv(meta_path)
if LABELS is None:
    ID_TO_LABEL = dict(zip(df_meta["target"], df_meta["category"]))
else:
    ID_TO_LABEL = {i: name for i, name in enumerate(LABELS)}

# تجهيز قائمة بأصوات الاختبار (Fold 5) بأرقام مجهولة بدون كشف الفئة
test_df = df_meta[df_meta["fold"] == 5].drop_duplicates("category").head(20)
SAMPLE_CHOICES = {}
for idx, (_, row) in enumerate(test_df.iterrows(), start=1):
    p = os.path.join(BASE_DIR, "audio", row["filename"])
    if os.path.exists(p):
        label_str = f"Test Sample #{idx:02d} — [{row['filename']}]"
        SAMPLE_CHOICES[label_str] = p

# تحميل الموديل وتحميل أوزان الـ classifier من داخل ckpt['classifier']
extractor = ASTFeatureExtractor.from_pretrained(MODEL_ID)
model = ASTForAudioClassification.from_pretrained(
    MODEL_ID, num_labels=50, ignore_mismatched_sizes=True
).to(DEVICE)

model.classifier.load_state_dict(ckpt["classifier"])
model.eval()
print("✅ Successfully loaded trained AST classifier (95.83% Test Accuracy)!")

def add_noise(waveform, snr_db):
    if snr_db >= 40:
        return waveform
    noise = torch.randn_like(waveform)
    sig_power = waveform.pow(2).mean().clamp(min=1e-12)
    noise_power = noise.pow(2).mean().clamp(min=1e-12)
    scale = torch.sqrt(sig_power / (noise_power * (10 ** (snr_db / 10.0))))
    return waveform + scale * noise

def predict_sound(selected_sample, mic_audio, snr_db):
    if mic_audio is not None:
        sr, audio_np = mic_audio
        audio_np = audio_np.astype(np.float32)
        if np.abs(audio_np).max() > 1.0:
            audio_np = audio_np / 32768.0
    elif selected_sample in SAMPLE_CHOICES:
        sr, audio_np = wavfile.read(SAMPLE_CHOICES[selected_sample])
        if audio_np.dtype == np.int16:
            audio_np = audio_np.astype(np.float32) / 32768.0
        elif audio_np.dtype == np.int32:
            audio_np = audio_np.astype(np.float32) / 2147483648.0
    else:
        return {"Select a sound from the list or record via mic": 1.0}, None

    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=-1)

    wav = torch.from_numpy(audio_np).unsqueeze(0)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)

    max_len = 16000 * 5
    if wav.shape[1] < max_len:
        wav = torch.nn.functional.pad(wav, (0, max_len - wav.shape[1]))
    else:
        wav = wav[:, :max_len]

    noisy_wav = add_noise(wav, snr_db)

    inputs = extractor(noisy_wav.squeeze(0).numpy(), sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs["input_values"].to(DEVICE)).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    top5_idx = np.argsort(probs)[::-1][:5]
    top5_dict = {ID_TO_LABEL.get(int(i), f"Class {i}"): float(probs[i]) for i in top5_idx}

    out_audio = (np.clip(noisy_wav.squeeze(0).numpy(), -1.0, 1.0) * 32767).astype(np.int16)
    return top5_dict, (16000, out_audio)

demo = gr.Interface(
    fn=predict_sound,
    inputs=[
        gr.Dropdown(choices=list(SAMPLE_CHOICES.keys()), value=list(SAMPLE_CHOICES.keys())[0] if SAMPLE_CHOICES else None, label="1. Select Unlabeled Test Audio (Fold 5)"),
        gr.Audio(sources=["microphone"], type="numpy", label="OR Record Live via Microphone (Leave empty to use selected sample)"),
        gr.Slider(minimum=-5, maximum=40, step=5, value=40, label="2. Noise Level SNR (dB) — [40 = Clean | 0 = Heavy | -5 = Extreme Noise]")
    ],
    outputs=[
        gr.Label(num_top_classes=5, label="Model Prediction — Top-5 Classes (AST - 95.83% Test Acc)"),
        gr.Audio(label="Audio Input (Listen to verify sound & noise level)")
    ],
    title="🔊 Environmental Sound Classification Under Noise (ESC-50)",
    description="Live Blind Demo — Deep Learning Final Project (Pharos University in Alexandria)."
)

if __name__ == "__main__":
    demo.launch(inbrowser=True)