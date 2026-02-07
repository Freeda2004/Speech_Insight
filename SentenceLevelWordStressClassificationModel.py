import torch
import torch.nn as nn
import torch.nn.functional as F


class WordLevelClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_layers=1, dropout=0.2, num_classes=2):
        """
        input_dim: should match feature_dim from your dataloader
                   (e.g. 768 for 1 layer, 1536 for 2 layers, 2304 for 3 layers, etc.)
        """
        super().__init__()
        self.bilstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.word_classifier = nn.Linear(2 * hidden_dim, num_classes)

    def forward(self, word_feats, word_lengths):
        # Pack sequence to handle variable word lengths
        packed_input = nn.utils.rnn.pack_padded_sequence(
            word_feats, word_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.bilstm(packed_input)

        # Unpack back to padded sequence
        hidden_states, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Word-level logits
        word_logits = self.word_classifier(hidden_states)  # [B, T_words, num_classes]

        return word_logits


def masked_cross_entropy(logits, targets, mask):
    """
    logits: [B, T, C]
    targets: [B, T]
    mask: [B, T] (True = valid word)
    """
    B, T, C = logits.shape
    logits_flat = logits.view(-1, C)
    targets_flat = targets.view(-1)
    mask_flat = mask.view(-1).bool()

    if mask_flat.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    logits_masked = logits_flat[mask_flat]
    targets_masked = targets_flat[mask_flat]

    return F.cross_entropy(logits_masked, targets_masked)
