# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import whisper
# import tempfile
# import os
# import torch
# from transformers import Wav2Vec2Processor, Wav2Vec2Model

# # -------------------------------------------------
# # App setup
# # -------------------------------------------------
# app = Flask(__name__)
# CORS(app)

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # -------------------------------------------------
# # Load Whisper (Punctuation Lab)
# # -------------------------------------------------
# print("Loading Whisper model...")
# whisper_model = whisper.load_model("small")

# # -------------------------------------------------
# # Load Stress Model resources (Stress Studio)
# # -------------------------------------------------
# print("Loading Word Stress model...")

# # Wav2Vec (used later when you plug real pipeline)
# processor = Wav2Vec2Processor.from_pretrained(
#     "facebook/wav2vec2-base-960h"
# )
# wav2vec_model = Wav2Vec2Model.from_pretrained(
#     "facebook/wav2vec2-base-960h"
# ).to(DEVICE)
# wav2vec_model.eval()

# # Stress classifier
# from SentenceLevelWordStressClassificationModel import WordLevelClassifier

# SELECTED_LAYERS = [9]
# INPUT_DIM = 768 * len(SELECTED_LAYERS)

# stress_model = WordLevelClassifier(input_dim=INPUT_DIM)
# stress_model.load_state_dict(
#     torch.load("word_level_model.pth", map_location=DEVICE)
# )
# stress_model.to(DEVICE)
# stress_model.eval()

# print("All models loaded successfully.")

# # -------------------------------------------------
# # Helper: Word stress inference (TEMP placeholder)
# # -------------------------------------------------
# def run_word_stress(audio_path):
#     """
#     TEMPORARY placeholder inference.
#     Replace this later with your real
#     WordLevelWav2VecDataset + timestamps.
#     """

#     # Dummy words + timestamps (for UI verification)
#     words = ["phonology", "analysis", "speech", "insight"]
#     timestamps = [(0.5, 0.9), (0.9, 1.3), (1.3, 1.7), (1.7, 2.2)]

#     # Fake features (shape: [1, num_words, input_dim])
#     feats = torch.randn(1, len(words), INPUT_DIM).to(DEVICE)
#     lengths = torch.tensor([len(words)]).to(DEVICE)

#     with torch.no_grad():
#         logits = stress_model(feats, lengths)
#         preds = logits.argmax(dim=-1)[0].cpu().tolist()

#     output = []
#     for i, w in enumerate(words):
#         output.append({
#             "word": w,
#             "index": i,
#             "stress": int(preds[i]),
#             "start": timestamps[i][0],
#             "end": timestamps[i][1]
#         })

#     return output

# # -------------------------------------------------
# # Route: Punctuation Lab (UNCHANGED)
# # -------------------------------------------------
# @app.route("/punctuate", methods=["POST"])
# def punctuate():
#     audio = request.files.get("audio")
#     text = request.form.get("text")  # accepted but not used

#     if audio is None:
#         return jsonify({"error": "No audio file"}), 400

#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
#         audio.save(tmp.name)
#         audio_path = tmp.name

#     result = whisper_model.transcribe(audio_path)

#     os.remove(audio_path)

#     return jsonify({
#         "punctuated_text": result["text"]
#     })

# # -------------------------------------------------
# # Route: Stress Studio (NEW)
# # -------------------------------------------------
# @app.route("/stress", methods=["POST"])
# def stress():
#     audio = request.files.get("audio")

#     if audio is None:
#         return jsonify({"error": "No audio file"}), 400

#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
#         audio.save(tmp.name)
#         audio_path = tmp.name

#     try:
#         stress_output = run_word_stress(audio_path)
#     finally:
#         os.remove(audio_path)

#     return jsonify({
#         "words": stress_output
#     })

# # -------------------------------------------------
# # Main
# # -------------------------------------------------
# if __name__ == "__main__":
#     app.run(debug=True)



# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import torch
# import torchaudio
# import whisper
# import tempfile
# import os

# from transformers import Wav2Vec2Processor, Wav2Vec2Model
# from SentenceLevelWordStressClassificationModel import WordLevelClassifier

# # ================= CONFIG =================
# MODEL_PATH = "word_level_model.pth"
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# SELECTED_LAYERS = [9]
# INPUT_DIM = 768 * len(SELECTED_LAYERS)
# # =========================================

# app = Flask(__name__)
# CORS(app)

# # ---------- LOAD MODELS ----------
# print("🔹 Loading Whisper...")
# whisper_model = whisper.load_model("base")

# print("🔹 Loading Wav2Vec2...")
# processor = Wav2Vec2Processor.from_pretrained(
#     "facebook/wav2vec2-base-960h"
# )
# wav2vec = Wav2Vec2Model.from_pretrained(
#     "facebook/wav2vec2-base-960h"
# ).to(DEVICE)
# wav2vec.eval()

# print("🔹 Loading Word Stress Model...")
# stress_model = WordLevelClassifier(input_dim=INPUT_DIM)
# stress_model.load_state_dict(
#     torch.load(MODEL_PATH, map_location=DEVICE)
# )
# stress_model.to(DEVICE)
# stress_model.eval()


# # ---------- FEATURE EXTRACTION ----------
# def extract_word_features(waveform, sr, word_timestamps):
#     inputs = processor(
#         waveform.squeeze().numpy(),
#         sampling_rate=sr,
#         return_tensors="pt"
#     ).input_values.to(DEVICE)

#     with torch.no_grad():
#         outputs = wav2vec(inputs, output_hidden_states=True)
#         hidden = outputs.hidden_states[SELECTED_LAYERS[0]]

#     total_audio_time = waveform.shape[1] / sr
#     T = hidden.shape[1]

#     feats = []
#     for w in word_timestamps:
#         start = int((w["start"] / total_audio_time) * T)
#         end = int((w["end"] / total_audio_time) * T)
#         end = max(end, start + 1)

#         feat = hidden[:, start:end, :].mean(dim=1)
#         feats.append(feat)

#     return torch.cat(feats, dim=0)


# # ==========================================================
# #  PUNCTUATION ENDPOINT (UNCHANGED)
# # ==========================================================
# @app.route("/punctuate", methods=["POST"])
# def punctuate():
#     if "audio" not in request.files or "text" not in request.form:
#         return jsonify({"error": "Missing audio or text"}), 400

#     audio = request.files["audio"]
#     raw_text = request.form["text"]

#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
#         audio.save(tmp.name)
#         audio_path = tmp.name

#     result = whisper_model.transcribe(audio_path)
#     os.remove(audio_path)

#     # ⚠️ KEEP YOUR EXISTING LOGIC HERE
#     # Placeholder (since you said punctuation already works)
#     punctuated = result["text"]

#     return jsonify({
#         "punctuated_text": punctuated
#     })


# # ==========================================================
# #  WORD STRESS ENDPOINT (NEW + FRONTEND-COMPATIBLE)
# # ==========================================================
# @app.route("/stress", methods=["POST"])
# def stress():
#     if "audio" not in request.files:
#         return jsonify({"error": "No audio file"}), 400

#     audio = request.files["audio"]

#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
#         audio.save(tmp.name)
#         audio_path = tmp.name

#     # Whisper word timestamps
#     result = whisper_model.transcribe(
#         audio_path,
#         word_timestamps=True
#     )

#     word_ts = []
#     for seg in result["segments"]:
#         for w in seg["words"]:
#             word_ts.append({
#                 "word": w["word"].strip(),
#                 "start": w["start"],
#                 "end": w["end"]
#             })

#     if not word_ts:
#         os.remove(audio_path)
#         return jsonify({"words": []})

#     waveform, sr = torchaudio.load(audio_path)

#     feats = extract_word_features(waveform, sr, word_ts)
#     lengths = torch.tensor([feats.shape[0]]).to(DEVICE)

#     with torch.no_grad():
#         logits = stress_model(feats.unsqueeze(0), lengths)
#         preds = logits.argmax(dim=-1).squeeze(0).cpu().numpy()

#     words = []
#     for i, p in enumerate(preds):
#         words.append({
#         "word": word_ts[i]["word"],
#         "stress": int(p),
#         "start": round(word_ts[i]["start"], 2),
#         "end": round(word_ts[i]["end"], 2)
#     })


#     os.remove(audio_path)

#     return jsonify({"words": words})


# # ---------- RUN ----------
# if __name__ == "__main__":
#     app.run(debug=True)



from flask import Flask, request, jsonify
from flask_cors import CORS
import keras
import torch
import torchaudio
import whisper
import tempfile
import os
import numpy as np
from flask import render_template
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from SentenceLevelWordStressClassificationModel import WordLevelClassifier

# ================= CONFIG =================
MODEL_PATH = "word_level_model.pth"

# 🔥 FORCE CPU (Flask-safe, fixes device error)
DEVICE = torch.device("cpu")

SELECTED_LAYERS = [9]
INPUT_DIM = 768 * len(SELECTED_LAYERS)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SYL_MODEL_PATH = os.path.join(BASE_DIR, "sylnet_gercorrect_new.keras")
HUBERT_NPY_DIR = os.path.join(BASE_DIR, "nppy_germ_folder")

# -------- SYLLABLE CONFIG --------
MAX_TIME = 200
SAMPLE_RATE = 16000
# =========================================



app = Flask(__name__)
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    supports_credentials=True
)


# ---------- LOAD MODELS ----------
print("🔹 Loading Whisper...")
whisper_model = whisper.load_model("base")

print("🔹 Loading Wav2Vec2...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-base-960h"
).to(DEVICE)
wav2vec.eval()

print("🔹 Loading Word Stress Model...")
stress_model = WordLevelClassifier(input_dim=INPUT_DIM)
stress_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
stress_model.to(DEVICE)
stress_model.eval()

print("🔹 Loading Syllable Model...")
syllable_model = keras.models.load_model(SYL_MODEL_PATH)

# 🔥 MOVE EMBEDDING TO SAME DEVICE
quant_embedder = torch.nn.Embedding(
    num_embeddings=1024,
    embedding_dim=512
).to(DEVICE)

# ==========================================================
#  FEATURE EXTRACTION — WORD STRESS
# ==========================================================
def extract_word_features(waveform, sr, word_timestamps):
    inputs = processor(
        waveform.squeeze().numpy(),
        sampling_rate=sr,
        return_tensors="pt"
    ).input_values.to(DEVICE)

    with torch.no_grad():
        outputs = wav2vec(inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[SELECTED_LAYERS[0]]

    total_audio_time = waveform.shape[1] / sr
    T = hidden.shape[1]

    feats = []
    for w in word_timestamps:
        start = int((w["start"] / total_audio_time) * T)
        end = int((w["end"] / total_audio_time) * T)
        end = max(end, start + 1)
        feats.append(hidden[:, start:end, :].mean(dim=1))

    return torch.cat(feats, dim=0)

# ==========================================================
#  FEATURE EXTRACTION — SYLLABLE (PRECOMPUTED NPY ✔)
# ==========================================================
def extract_syllable_features(wav_path):
    waveform, sr = torchaudio.load(wav_path)

    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)

    inputs = processor(
        waveform.squeeze(),
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    ).input_values.to(DEVICE)

    with torch.no_grad():
        wav_feats = wav2vec(inputs).last_hidden_state.squeeze(0).to(DEVICE)
        # (T, 768)

    npy_path = os.path.join(
        HUBERT_NPY_DIR,
        os.path.basename(wav_path).replace(".wav", ".npy")
    )

    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"HuBERT file not found: {npy_path}")

    indices = torch.tensor(
        np.load(npy_path),
        dtype=torch.long
    ).to(DEVICE)

    with torch.no_grad():
        hubert_feats = quant_embedder(indices)
        # (T, 512)

    T = min(wav_feats.shape[0], hubert_feats.shape[0])
    combined = torch.cat([wav_feats[:T], hubert_feats[:T]], dim=-1)

    if combined.shape[0] > MAX_TIME:
        combined = combined[:MAX_TIME]
    else:
        pad = MAX_TIME - combined.shape[0]
        combined = torch.nn.functional.pad(combined, (0, 0, 0, pad))

    return combined.cpu().numpy()

# ==========================================================
#  PUNCTUATION ENDPOINT
# ==========================================================
@app.route("/punctuate", methods=["POST"])
def punctuate():
    audio = request.files["audio"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        audio.save(tmp.name)
        audio_path = tmp.name

    result = whisper_model.transcribe(audio_path)
    os.remove(audio_path)

    return jsonify({"punctuated_text": result["text"]})

# ==========================================================
#  WORD STRESS ENDPOINT
# ==========================================================
@app.route("/stress", methods=["POST"])
def stress():
    audio = request.files["audio"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        audio.save(tmp.name)
        audio_path = tmp.name

    result = whisper_model.transcribe(audio_path, word_timestamps=True)

    word_ts = []
    for seg in result["segments"]:
        for w in seg["words"]:
            word_ts.append({
                "word": w["word"].strip(),
                "start": w["start"],
                "end": w["end"]
            })

    waveform, sr = torchaudio.load(audio_path)
    feats = extract_word_features(waveform, sr, word_ts)
    lengths = torch.tensor([feats.shape[0]]).to(DEVICE)

    with torch.no_grad():
        preds = stress_model(feats.unsqueeze(0), lengths).argmax(dim=-1)[0]

    os.remove(audio_path)

    return jsonify({
        "words": [
            {
                "word": word_ts[i]["word"],
                "stress": int(preds[i]),
                "start": round(word_ts[i]["start"], 2),
                "end": round(word_ts[i]["end"], 2)
            }
            for i in range(len(word_ts))
        ]
    })

# ==========================================================
#  SYLLABLE PROFILING ENDPOINT
# ==========================================================
@app.route("/syllables", methods=["POST"])
def syllables():
    if "audio" not in request.files:
        return jsonify({"error": "No audio uploaded"}), 400

    audio = request.files["audio"]

    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    audio_path = os.path.join(upload_dir, audio.filename)
    audio.save(audio_path)

    # 🔴 EARLY CHECK FOR NPY
    npy_path = os.path.join(
        HUBERT_NPY_DIR,
        audio.filename.replace(".wav", ".npy")
    )

    if not os.path.exists(npy_path):
        return jsonify({
            "error": f"No precomputed HuBERT file for {audio.filename}"
        }), 400

    try:
        waveform, sr = torchaudio.load(audio_path)
        duration = waveform.shape[1] / sr

        feats = extract_syllable_features(audio_path)
        feats = np.expand_dims(feats, axis=0).astype(np.float32)

        pred = syllable_model.predict(feats, verbose=0)
        syllable_count = int(round(pred[0][0]))

        # ✅ SAFE RATE COMPUTATION
        syllable_rate = (
            round(syllable_count / duration, 2)
            if duration > 0 else 0.0
        )

    except Exception as e:
        print("❌ Syllable error:", e)
        return jsonify({"error": str(e)}), 500
    print("✅ Sending syllable response")

    return jsonify({
        "syllable_count": syllable_count,
        "syllable_rate": syllable_rate,
        "duration": round(duration, 2)
    })
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/app")
def app_page():
    return render_template("index.html")

# ---------- RUN ----------
if __name__ == "__main__":
      app.run(host="0.0.0.0", port=5000)
