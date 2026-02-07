import os
import torch
import torchaudio
import scipy.io as sio
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import Wav2Vec2Processor, Wav2Vec2Model

torchaudio.set_audio_backend("soundfile")


def load_transcriptions(trans_file):
    transcriptions = {}
    with open(trans_file, "r") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                file_id, sentence = parts
                transcriptions[file_id] = sentence.split()
    return transcriptions


def load_labels(label_file, split):
    data = {}
    with open(label_file, "r") as f:
        for line in f:
            file_id, labels_str = line.strip().split(maxsplit=1)
            labels = eval(labels_str)
            data[file_id] = {"labels": labels, "split": split}
    return data


class WordLevelWav2VecDataset(Dataset):
    def __init__(self, file_ids, transcriptions, labels, wav_dir, ts_dir,
                 processor, model, sample_rate=16000, layer_indices=None):
        self.file_ids = file_ids
        self.transcriptions = transcriptions
        self.labels = labels
        self.wav_dir = wav_dir
        self.ts_dir = ts_dir
        self.processor = processor
        self.model = model
        self.sample_rate = sample_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.layer_indices = layer_indices if layer_indices is not None else [-1]

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        path = os.path.join(self.wav_dir, f"{file_id}.wav")

        # ====== 1. Load audio ======
        waveform, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        waveform = waveform.squeeze(0)

        # ====== 2. Extract wav2vec features ======
        input_values = self.processor(
            waveform, return_tensors="pt", sampling_rate=self.sample_rate
        ).input_values.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_values, output_hidden_states=True)
            selected_features = [outputs.hidden_states[i].squeeze(0) for i in self.layer_indices]

        duration = waveform.shape[0] / self.sample_rate
        num_frames = selected_features[0].shape[0]

        def time_to_frame(time):
            return int(time / duration * num_frames)

        # ====== 3. Word-level features ======
        word_feats, word_labels = [], []

        ts_txt = os.path.join(self.ts_dir, f"{file_id}.txt")
        ts_mat = os.path.join(self.ts_dir, f"{file_id}.mat")

        if os.path.exists(ts_txt):
            # Case 1: TXT timestamps
            with open(ts_txt, "r") as f:
                ts_lines = f.readlines()
            word_times = [(float(start), float(end), word)
                          for start, end, word in (line.strip().split() for line in ts_lines)]

        elif os.path.exists(ts_mat):
            # Case 2: MAT timestamps
            mat_data = sio.loadmat(ts_mat)
            words = mat_data["words"]
            times = mat_data["spurtWordTimes"]
            word_times = [(float(times[i, 0]), float(times[i, 1]), str(words[i][0][0]).upper())
                          for i in range(len(words))]

        else:
            raise FileNotFoundError(f"No .txt or .mat timestamp for {file_id}")

        labels = self.labels[file_id]["labels"]

        for i, (start, end, word) in enumerate(word_times):
            start_idx, end_idx = time_to_frame(start), time_to_frame(end)

            word_feat_layers = [
                layer_features[start_idx:end_idx + 1].mean(dim=0)
                for layer_features in selected_features
            ]
            word_concat_feat = torch.cat(word_feat_layers, dim=-1)
            word_feats.append(word_concat_feat)

            if i < len(labels):
                word_labels.append(int(labels[i]))
            else:
                word_labels.append(0)  # default if mismatch

        word_feats = torch.stack(word_feats)
        word_labels = torch.tensor(word_labels, dtype=torch.long)

        return {
            "word_feats": word_feats,
            "word_labels": word_labels,
            "file_name": file_id,
        }


def word_collate_fn(batch):
    word_feats = [item['word_feats'] for item in batch]
    word_labels = [item['word_labels'] for item in batch]
    file_names = [item['file_name'] for item in batch]

    padded_word_feats = pad_sequence(word_feats, batch_first=True)
    padded_word_labels = pad_sequence(word_labels, batch_first=True, padding_value=-100)

    word_lengths = torch.tensor([len(w) for w in word_feats])
    max_word_len = padded_word_labels.size(1)
    word_mask = torch.arange(max_word_len).unsqueeze(0) < word_lengths.unsqueeze(1)

    return {
        "word_feats": padded_word_feats,
        "word_labels": padded_word_labels,
        "word_lengths": word_lengths,
        "word_mask": word_mask,
        "file_names": file_names,
    }
