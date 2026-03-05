"""
Intel Model Components
=======================

- IntelFeatureExtractor: OpenVINO encoder wrapper
- EncoderMLP:  Feedforward decoder — static shapes → compiles on OpenVINO GPU ✓  (default)
- EncoderLSTM: Bidirectional LSTM with attention — dynamic shapes → CPU-only at inference ✗ (legacy)
"""

import numpy as np
import torch
import torch.nn as nn
from openvino.runtime import Core


# =============================
# OpenVINO Feature Extractor
# =============================
class IntelFeatureExtractor:
    """Wraps the Intel action-recognition-0001 encoder (OpenVINO)."""

    def __init__(self, encoder_xml, encoder_bin):
        ie = Core()
        model = ie.read_model(model=encoder_xml, weights=encoder_bin)
        self.encoder = ie.compile_model(model, device_name="CPU")

        inp = self.encoder.inputs[0]
        self.input_name = inp.get_any_name()
        self.input_shape = list(inp.get_shape())
        print(f"✓ Encoder input: {self.input_name}, shape: {self.input_shape}")

    def encode(self, frames_batch):
        """
        Encode a batch of frame sequences.

        Args:
            frames_batch: (B, T, C, H, W) tensor or ndarray

        Returns:
            torch.FloatTensor (B, T, feat_dim)
        """
        if isinstance(frames_batch, torch.Tensor):
            frames_batch = frames_batch.cpu().numpy()

        B, T, C, H, W = frames_batch.shape
        all_feats = []

        for b in range(B):
            seq_feats = []
            for t in range(T):
                frame = frames_batch[b, t]              # (C, H, W)
                frame = np.expand_dims(frame, axis=0)   # (1, C, H, W)
                frame = self._preprocess(frame)

                out = self.encoder([frame])
                try:
                    feat = out[self.encoder.output(0)]
                except Exception:
                    feat = list(out.values())[0]

                if feat.ndim > 1:
                    feat = feat.reshape(feat.shape[0], -1)
                seq_feats.append(feat)

            seq_feats = np.concatenate(seq_feats, axis=0)  # (T, feat_dim)
            all_feats.append(seq_feats)

        return torch.tensor(np.stack(all_feats, axis=0), dtype=torch.float32)

    @staticmethod
    def _preprocess(batch):
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
        batch = batch * 255.0
        batch = (batch / 255.0 - mean) / std
        return batch.astype(np.float32)


# =============================
# EncoderMLP  ← DEFAULT DECODER
# =============================
class EncoderMLP(nn.Module):
    """
    Feedforward classifier over a flattened feature sequence.

    Input:  (B, T, feat_dim)
    Output: logits (B, num_classes),  attn=None  (same return signature as EncoderLSTM)

    Why this instead of LSTM
    ------------------------
    LSTM → LSTMSequence ops with dynamic shapes in OpenVINO IR
         → GPU plugin rejects it → falls back to CPU at inference

    MLP  → only Linear + BatchNorm + ReLU + Dropout → fully static shapes
         → compiles on OpenVINO GPU with no issues
    """

    def __init__(self, feature_dim=512, hidden_dim=256, num_classes=31,
                 sequence_length=16, dropout=0.3, **kwargs):
        super().__init__()
        self.feature_dim     = feature_dim
        self.hidden_dim      = hidden_dim
        self.num_classes     = num_classes
        self.sequence_length = sequence_length

        flat_dim = sequence_length * feature_dim

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

    def forward(self, x):
        """
        Args:
            x: (B, T, feat_dim)
        Returns:
            logits: (B, num_classes)
            None:   placeholder — keeps the same return signature as EncoderLSTM
        """
        return self.net(x), None


# =============================
# EncoderLSTM  ← LEGACY (CPU-only at inference)
# =============================
class EncoderLSTM(nn.Module):
    """
    2-layer bidirectional LSTM with attention for action classification.

    ⚠️  OpenVINO GPU plugin does NOT support LSTMSequence ops with dynamic
        shapes. Models trained with this decoder will silently fall back to
        CPU at inference time. Use EncoderMLP instead for GPU inference.

    Kept here for:
      - loading old checkpoints
      - CPU-only deployments where LSTM accuracy matters

    Architecture:
        encoder features → BiLSTM(2) → LayerNorm → Attention → Classifier
    """

    def __init__(self, feature_dim=512, hidden_dim=256, num_classes=31,
                 num_layers=2, dropout=0.3, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        bidir_dim = hidden_dim * 2

        self.ln1 = nn.LayerNorm(bidir_dim)

        self.attention = nn.Sequential(
            nn.Linear(bidir_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self.ln2 = nn.LayerNorm(bidir_dim)
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(bidir_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                n = param.shape[0]
                param.data[n // 4 : n // 2].fill_(1.0)  # forget gate

        for layer in [self.attention, self.classifier]:
            for m in layer:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        m.bias.data.fill_(0.01)

    def forward(self, x):
        """
        Args:
            x: (B, T, feature_dim)
        Returns:
            logits (B, num_classes), attention_weights (B, T)
        """
        lstm_out, _ = self.lstm(x)
        lstm_out = self.ln1(lstm_out)

        attn_w = self.attention(lstm_out)
        attn_w = torch.softmax(attn_w, dim=1)

        context = torch.sum(lstm_out * attn_w, dim=1)
        context = self.ln2(context)
        context = self.dropout(context)

        logits = self.classifier(context)
        return logits, attn_w.squeeze(-1)


# =============================
# Factory
# =============================
def build_decoder(decoder_type, feature_dim, hidden_dim, num_classes,
                  sequence_length=16, num_layers=2, dropout=0.3):
    """
    Build a decoder by name.

    Args:
        decoder_type: "mlp" (default, GPU-compatible) or "lstm" (CPU-only at inference)
    """
    decoder_type = decoder_type.lower()

    if decoder_type == "mlp":
        model = EncoderMLP(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            sequence_length=sequence_length,
            dropout=dropout,
        )
        print(f"✅ Decoder: EncoderMLP (GPU-compatible at inference)")
    elif decoder_type == "lstm":
        model = EncoderLSTM(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_layers=num_layers,
            dropout=dropout,
        )
        print(f"⚠️  Decoder: EncoderLSTM (will fall back to CPU at inference — consider 'mlp')")
    else:
        raise ValueError(f"Unknown decoder_type '{decoder_type}'. Use 'mlp' or 'lstm'.")

    total = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total:,}")
    return model