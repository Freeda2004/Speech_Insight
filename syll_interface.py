

import os, numpy as np, torch, torchaudio
import tensorflow as tf
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# ---------- CONFIG ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "trained_models",
    "sylnet_gercorrect_new.keras"
)

HUBERT_NPY_DIR = os.path.join(
    BASE_DIR,
    "trained_models",
    "nppy_germ_folder"
)

MAX_TIME = 200
SAMPLE_RATE = 16000

# ---------- LOAD MODELS ----------
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec_model.eval()

quant_embedder = torch.nn.Embedding(
    num_embeddings=1024,
    embedding_dim=512
)

model = tf.keras.models.load_model(MODEL_PATH)

# ---------- FEATURE EXTRACTION (IDENTICAL LOGIC) ----------
def extract_features(wav_path):
    waveform, sr = torchaudio.load(wav_path)

    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)

    input_values = processor(
        waveform.squeeze(),
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    ).input_values

    with torch.no_grad():
        wav2vec_feats = wav2vec_model(input_values).last_hidden_state.squeeze(0)
        # (T, 768)

    # ---- Load HuBERT indices (EXACTLY like eval code) ----
    npy_path = os.path.join(
        HUBERT_NPY_DIR,
        os.path.basename(wav_path).replace(".wav", ".npy")
    )

    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"HuBERT file not found: {npy_path}")

    indices = np.load(npy_path)
    frame_indices = torch.tensor(indices, dtype=torch.long)

    with torch.no_grad():
        symbolic = quant_embedder(frame_indices)
        # (T, 512)

    # ---- Temporal alignment ----
    T = min(wav2vec_feats.shape[0], symbolic.shape[0])
    combined = torch.cat(
        [wav2vec_feats[:T], symbolic[:T]],
        dim=-1
    )
    # (T, 1280)

    # ---- Pad / truncate to MAX_TIME ----
    if combined.shape[0] > MAX_TIME:
        combined = combined[:MAX_TIME]
    else:
        pad = MAX_TIME - combined.shape[0]
        combined = torch.nn.functional.pad(
            combined,
            (0, 0, 0, pad)
        )

    return combined.detach().cpu().numpy()
    # (200, 1280)

# ---------- INFERENCE ----------
def predict_syllables(wav_path, mean=None, std=None):
    feat = extract_features(wav_path)

    # Normalization (same logic as evaluation)
    if mean is not None and std is not None:
        feat = (feat - mean) / (std + 1e-6)

    feat = np.expand_dims(feat, axis=0)  # (1, 200, 1280)

    pred = model.predict(feat, verbose=0)
    return int(round(pred[0][0]))