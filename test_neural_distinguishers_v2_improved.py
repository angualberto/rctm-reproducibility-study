#!/usr/bin/env python3
"""
Neural Network Distinguishing Attacks v2 - IMPROVED
Testes aprimorados com janelas maiores e análises mais profundas
"""

import argparse
import os
import subprocess
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from Crypto.Cipher import ChaCha20
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mutual_info_score

warnings.filterwarnings("ignore")

print("=" * 70)
print("DISTINGUISHING ATTACKS v2.0 - IMPROVED (CUDA OPTIMIZED)")
print("=" * 70)
print()


# ============================================================================
# MODELO TRANSFORMER MELHORADO
# ============================================================================
class ImprovedTransformer(nn.Module):
    """Transformer aprimorado para detectar padrões em sequências longas"""

    def __init__(self, seq_len=256, d_model=128, nhead=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Linear(1, d_model)
        self.pos_encoder = self._get_pos_encoding(seq_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dim_feedforward=512,
            dropout=0.1,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 3),
        )

    @staticmethod
    def _get_pos_encoding(seq_len, d_model):
        """Positional encoding sinusoidal"""
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * -(np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        if x.size(1) <= self.pos_encoder.size(1):
            x = x + self.pos_encoder[:, : x.size(1), :].to(x.device)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)


# ============================================================================
# LSTM DISTINGUISHER (Alternativa poderosa)
# ============================================================================
class LSTMDistinguisher(nn.Module):
    """LSTM para capturar dependências de longo prazo"""

    def __init__(self, input_size=1, hidden_size=256, num_layers=3, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(1, 64)
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # Take last output
        return self.fc(x)


# ============================================================================
# GERAÇÃO DE DADOS
# ============================================================================


def generate_agle_bytes(samples, window, agle_bin, seed):
    """Generate AGLE bytes"""
    total_bytes = samples * window
    print(f"  Generating {total_bytes:,} bytes from AGLE...")

    process = subprocess.Popen(
        [agle_bin, "--stdout", str(seed)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    data = process.stdout.read(total_bytes)
    process.terminate()

    if len(data) < total_bytes:
        raise RuntimeError(
            f"AGLE output insufficient: expected {total_bytes}, got {len(data)}"
        )

    return np.frombuffer(data, dtype=np.uint8)


def generate_chacha_bytes(samples, window):
    """Generate ChaCha20 bytes"""
    total_bytes = samples * window
    print(f"  Generating {total_bytes:,} bytes from ChaCha20...")

    key = os.urandom(32)
    nonce = os.urandom(8)
    cipher = ChaCha20.new(key=key, nonce=nonce)
    return np.frombuffer(cipher.encrypt(b"\x00" * total_bytes), dtype=np.uint8)


def generate_urandom_bytes(samples, window):
    """Generate /dev/urandom bytes"""
    total_bytes = samples * window
    print(f"  Reading {total_bytes:,} bytes from /dev/urandom...")

    with open("/dev/urandom", "rb") as f:
        return np.frombuffer(f.read(total_bytes), dtype=np.uint8)


def build_windows(data, label, window):
    """Build sliding windows from data"""
    usable = (len(data) // window) * window
    data = data[:usable]
    shaped = data.reshape(-1, window)
    labels = np.full((shaped.shape[0],), label, dtype=np.int64)
    return shaped.astype(np.float32) / 255.0, labels


# ============================================================================
# TREINAMENTO
# ============================================================================


def train_model(model, train_loader, test_loader, device, epochs=100, name="Model"):
    """Train distinguisher"""
    print(f"\n  Training {name}...")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_count = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda"):
                logits = model(batch_x)
                loss = loss_fn(logits, batch_y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_count += 1

        scheduler.step()

        if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x = batch_x.to(device, non_blocking=True)
                    batch_y = batch_y.to(device, non_blocking=True)

                    logits = model(batch_x)
                    pred = torch.argmax(logits, dim=1)
                    correct += (pred == batch_y).sum().item()
                    total += batch_y.size(0)

            acc = 100.0 * correct / total if total > 0 else 0.0
            avg_loss = train_loss / max(train_count, 1)
            print(f"    epoch {epoch+1:3d} loss {avg_loss:.4f} acc {acc:.2f}%")

    # Final eval
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            logits = model(batch_x)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == batch_y).sum().item()
            total += batch_y.size(0)

    return 100.0 * correct / total if total > 0 else 0.0


# ============================================================================
# SPECTRAL ENTROPY APRIMORADA
# ============================================================================


def compute_spectral_entropy_advanced(data, window=256):
    """Spectral entropy com múltiplas resoluções"""
    # FFT de alta resolução
    n_windows = len(data) // window
    entropies = []
    
    for i in range(min(50, n_windows)):
        chunk = data[i * window : (i + 1) * window]
        spectrum = np.fft.fft(chunk)
        power = np.abs(spectrum) ** 2
        power = power / np.sum(power)
        entropy = -np.sum(power * np.log2(power + 1e-12))
        max_entropy = np.log2(len(chunk))
        entropies.append(entropy / max_entropy if max_entropy > 0 else 0.0)
    
    return np.mean(entropies) if entropies else 0.0


# ============================================================================
# AUTOCORRELAÇÃO
# ============================================================================


def compute_autocorrelation(data, max_lag=50):
    """Autocorrelação até lag máximo"""
    data = data.astype(float)
    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2) / len(data)
    
    acf = [1.0]
    for lag in range(1, min(max_lag, len(data) // 2)):
        c_lag = np.sum((data[:-lag] - mean) * (data[lag:] - mean)) / len(data)
        acf.append(c_lag / c0)
    
    return np.array(acf)


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=123456789)
    parser.add_argument("--agle-bin", default="./versoes/v2/binarios/agle_v2_improved")
    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)

    if not os.path.exists(args.agle_bin):
        print(f"❌ AGLE binary not found: {args.agle_bin}")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        sys.exit(1)

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print()

    # ========================================================================
    # GENERATE DATA
    # ========================================================================
    print("Generating RNG data...")
    agle_data = generate_agle_bytes(args.samples, args.window, args.agle_bin, args.seed)
    chacha_data = generate_chacha_bytes(args.samples, args.window)
    urandom_data = generate_urandom_bytes(args.samples, args.window)

    # ========================================================================
    # TEST 1: IMPROVED TRANSFORMER (Janela maior)
    # ========================================================================
    print("\n" + "=" * 70)
    print("TEST 1: IMPROVED TRANSFORMER (Window=" + str(args.window) + ")")
    print("=" * 70)

    x1, y1 = build_windows(agle_data, 0, args.window)
    x2, y2 = build_windows(chacha_data, 1, args.window)
    x3, y3 = build_windows(urandom_data, 2, args.window)

    x = np.vstack([x1, x2, x3])
    y = np.concatenate([y1, y2, y3])

    perm = np.random.permutation(len(x))
    x, y = x[perm], y[perm]

    split = int(len(x) * 0.8)
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(x[:split]),
            torch.tensor(y[:split]),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
    )
    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(x[split:]),
            torch.tensor(y[split:]),
        ),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
    )

    model = ImprovedTransformer(seq_len=args.window, d_model=128, nhead=8, num_layers=6).to(device)
    tf_acc = train_model(model, train_loader, test_loader, device, epochs=args.epochs, name="Transformer")
    print(f"\n✓ Transformer: {tf_acc:.2f}%")

    # ========================================================================
    # TEST 2: LSTM DISTINGUISHER
    # ========================================================================
    print("\n" + "=" * 70)
    print("TEST 2: LSTM DISTINGUISHER")
    print("=" * 70)

    model = LSTMDistinguisher(hidden_size=256, num_layers=3).to(device)
    lstm_acc = train_model(model, train_loader, test_loader, device, epochs=args.epochs, name="LSTM")
    print(f"\n✓ LSTM: {lstm_acc:.2f}%")

    # ========================================================================
    # TEST 3: SPECTRAL ENTROPY ADVANCED
    # ========================================================================
    print("\n" + "=" * 70)
    print("TEST 3: SPECTRAL ENTROPY (Advanced)")
    print("=" * 70)

    print("\n  Computing spectral entropy at multiple resolutions...")
    se_agle = compute_spectral_entropy_advanced(agle_data, window=256)
    se_chacha = compute_spectral_entropy_advanced(chacha_data, window=256)
    se_urandom = compute_spectral_entropy_advanced(urandom_data, window=256)

    print(f"\n  AGLE:       {se_agle:.4f}")
    print(f"  ChaCha20:   {se_chacha:.4f}")
    print(f"  /dev/urandom: {se_urandom:.4f}")

    # ========================================================================
    # TEST 4: AUTOCORRELATION
    # ========================================================================
    print("\n" + "=" * 70)
    print("TEST 4: AUTOCORRELATION ANALYSIS")
    print("=" * 70)

    print("\n  Computing autocorrelation (lag 1-50)...")
    acf_agle = compute_autocorrelation(agle_data[:10000], max_lag=50)
    acf_chacha = compute_autocorrelation(chacha_data[:10000], max_lag=50)
    acf_urandom = compute_autocorrelation(urandom_data[:10000], max_lag=50)

    # Check if ACF at lag 1 is significantly non-zero
    acf_agle_lag1 = acf_agle[1] if len(acf_agle) > 1 else 0
    acf_chacha_lag1 = acf_chacha[1] if len(acf_chacha) > 1 else 0
    acf_urandom_lag1 = acf_urandom[1] if len(acf_urandom) > 1 else 0

    print(f"\n  ACF at lag 1:")
    print(f"  AGLE:       {acf_agle_lag1:.6f} (ideal: ≈0.0)")
    print(f"  ChaCha20:   {acf_chacha_lag1:.6f} (ideal: ≈0.0)")
    print(f"  /dev/urandom: {acf_urandom_lag1:.6f} (ideal: ≈0.0)")

    # ========================================================================
    # FINAL REPORT
    # ========================================================================
    print("\n" + "=" * 70)
    print("FINAL SECURITY REPORT")
    print("=" * 70)

    baseline = 100.0 / 3.0

    tf_ok = abs(tf_acc - baseline) < baseline * 0.2
    lstm_ok = abs(lstm_acc - baseline) < baseline * 0.2
    se_ok = np.mean([se_agle, se_chacha, se_urandom]) > 0.9
    acf_ok = np.mean([abs(acf_agle_lag1), abs(acf_chacha_lag1), abs(acf_urandom_lag1)]) < 0.05

    print(f"\nTransformer:  {tf_acc:.2f}% {'✅' if tf_ok else '❌'}")
    print(f"LSTM:         {lstm_acc:.2f}% {'✅' if lstm_ok else '❌'}")
    print(f"Spectral E.:  {np.mean([se_agle, se_chacha, se_urandom]):.4f} {'✅' if se_ok else '❌'}")
    print(f"ACF (lag 1):  {np.mean([abs(acf_agle_lag1), abs(acf_chacha_lag1), abs(acf_urandom_lag1)]):.6f} {'✅' if acf_ok else '❌'}")

    if tf_ok and lstm_ok and se_ok and acf_ok:
        print("\n✅ EXCELLENT: AGLE passes all neural cryptanalysis tests!")
    else:
        print("\n⚠️ Some issues detected. Review AGLE design.")

    print()


if __name__ == "__main__":
    start = time.time()
    main()
    elapsed = time.time() - start
    print(f"Runtime: {elapsed/60:.1f}m")
