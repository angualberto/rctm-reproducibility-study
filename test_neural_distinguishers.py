#!/usr/bin/env python3
"""
Neural Network Distinguishing Attacks against AGLE
Tests: Transformer, CNN, Next-Byte Predictor, Spectral Entropy, Mutual Information
"""

import argparse
import os
import subprocess
import sys
import time
import warnings
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Crypto.Cipher import ChaCha20
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mutual_info_score

warnings.filterwarnings("ignore")

print("=" * 60)
print("NEURAL DISTINGUISHING ATTACKS v2.0 (CUDA OPTIMIZED)")
print("=" * 60)
print()


# ============================================================================
# 1. TRANSFORMER DISTINGUISHER (Efficient Version)
# ============================================================================
class EfficientTransformer(nn.Module):
    """Optimized Transformer for CUDA"""

    def __init__(self, d_model=64, nhead=8, num_layers=4):
        super().__init__()
        self.embedding = nn.Linear(1, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 256, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dim_feedforward=256,
            dropout=0.05,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        x = x.unsqueeze(-1)  # (batch, window) -> (batch, window, 1)
        x = self.embedding(x)  # (batch, window, d_model)
        x = x + self.pos_encoder[:, : x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.fc(x)


# ============================================================================
# 2. CNN DISTINGUISHER (Efficient 1D Convolutions)
# ============================================================================
class CNNDistinguisher(nn.Module):
    """CNN for local pattern detection"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, window) -> (batch, 1, window)
        return self.net(x)


# ============================================================================
# 3. NEXT-BYTE PREDICTOR (Strong Cryptanalytic Test)
# ============================================================================
class NextBytePredictor(nn.Module):
    """Predicts next byte from previous 128 bytes"""

    def __init__(self, context_size=128):
        super().__init__()
        self.context_size = context_size
        self.net = nn.Sequential(
            nn.Linear(context_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================================
# DATA GENERATION
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
# TRAINING & EVALUATION
# ============================================================================


def train_distinguisher(
    model, train_loader, test_loader, device, epochs=50, name="Model"
):
    """Train a distinguisher model"""
    print(f"\n  Training {name}...")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0

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

        # Evaluate every N epochs
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
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
            best_acc = max(best_acc, acc)
            avg_loss = train_loss / max(train_count, 1)
            print(f"    epoch {epoch+1:3d} loss {avg_loss:.4f} acc {acc:.2f}%")

    return best_acc


def evaluate(model, loader, device):
    """Final evaluation"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            logits = model(batch_x)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == batch_y).sum().item()
            total += batch_y.size(0)

    return 100.0 * correct / total if total > 0 else 0.0


# ============================================================================
# SPECTRAL ENTROPY TEST
# ============================================================================


def spectral_entropy(data):
    """Compute spectral entropy of sequence"""
    spectrum = np.fft.fft(data)
    power = np.abs(spectrum) ** 2
    power = power / np.sum(power)
    max_entropy = np.log2(len(data))
    entropy = -np.sum(power * np.log2(power + 1e-12))
    return entropy / max_entropy if max_entropy > 0 else 0.0


def test_spectral_entropy(agle_data, chacha_data, urandom_data, window=128):
    """Test spectral entropy for all RNGs"""
    print("\n  Computing Spectral Entropy...")

    # Compute for each RNG (20 samples)
    agle_entropies = [
        spectral_entropy(agle_data[i * window : (i + 1) * window])
        for i in range(min(20, len(agle_data) // window))
    ]
    chacha_entropies = [
        spectral_entropy(chacha_data[i * window : (i + 1) * window])
        for i in range(min(20, len(chacha_data) // window))
    ]
    urandom_entropies = [
        spectral_entropy(urandom_data[i * window : (i + 1) * window])
        for i in range(min(20, len(urandom_data) // window))
    ]

    agle_mean = np.mean(agle_entropies)
    chacha_mean = np.mean(chacha_entropies)
    urandom_mean = np.mean(urandom_entropies)

    return {
        "agle": agle_mean,
        "chacha": chacha_mean,
        "urandom": urandom_mean,
    }


# ============================================================================
# MUTUAL INFORMATION TEST
# ============================================================================


def test_mutual_information(data, max_lag=10, max_samples=10000):
    """Test mutual information between consecutive bytes"""
    print("\n  Computing Mutual Information...")

    # Limit to max_samples for speed
    data = data[:max_samples]

    mi_scores = []
    for lag in range(1, min(max_lag + 1, len(data) // 2)):
        x = data[:-lag]
        y = data[lag:]
        mi = mutual_info_score(x, y)
        mi_scores.append(mi)

    return np.mean(mi_scores) if mi_scores else 0.0


# ============================================================================
# NEXT-BYTE PREDICTION TASK
# ============================================================================


def build_next_byte_dataset(data, context_size=128):
    """Build (context, target) pairs for next-byte prediction"""
    x = []
    y = []
    for i in range(len(data) - context_size):
        x.append(data[i : i + context_size])
        y.append(data[i + context_size])
    return np.array(x, dtype=np.float32) / 255.0, np.array(y, dtype=np.int64)


def train_next_byte_predictor(model, train_loader, test_loader, device, epochs=50):
    """Train next-byte predictor"""
    print(f"\n  Training Next-Byte Predictor...")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
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

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
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
            print(f"    epoch {epoch+1:3d} loss {train_loss/max(train_count,1):.4f} acc {acc:.2f}%")

    # Final evaluation
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
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=123456789)
    parser.add_argument(
        "--agle-bin", default="./versoes/v2/binarios/agle_v2_improved"
    )
    parser.add_argument("--context-size", type=int, default=128)
    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)

    if not os.path.exists(args.agle_bin):
        print(f"❌ AGLE binary not found: {args.agle_bin}")
        sys.exit(1)

    # Check CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Exiting.")
        sys.exit(1)

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print()

    # ========================================================================
    # GENERATE DATA
    # ========================================================================
    print("Step 1: Generating RNG data...")
    agle_data = generate_agle_bytes(args.samples, args.window, args.agle_bin, args.seed)
    chacha_data = generate_chacha_bytes(args.samples, args.window)
    urandom_data = generate_urandom_bytes(args.samples, args.window)

    # ========================================================================
    # TEST 1: TRANSFORMER DISTINGUISHER
    # ========================================================================
    print("\n" + "=" * 60)
    print("TEST 1: TRANSFORMER DISTINGUISHER")
    print("=" * 60)

    x1, y1 = build_windows(agle_data, 0, args.window)
    x2, y2 = build_windows(chacha_data, 1, args.window)
    x3, y3 = build_windows(urandom_data, 2, args.window)

    x = np.vstack([x1, x2, x3])
    y = np.concatenate([y1, y2, y3])

    perm = np.random.permutation(len(x))
    x, y = x[perm], y[perm]

    split = int(len(x) * 0.8)
    x_train, y_train = torch.tensor(x[:split]), torch.tensor(y[:split])
    x_test, y_test = torch.tensor(x[split:]), torch.tensor(y[split:])

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    print(f"Train samples: {len(x_train):,}")
    print(f"Test samples: {len(x_test):,}")
    print(f"Window: {args.window} bytes")

    model_transformer = EfficientTransformer(d_model=64, nhead=8, num_layers=4).to(
        device
    )
    transformer_acc = train_distinguisher(
        model_transformer,
        train_loader,
        test_loader,
        device,
        epochs=args.epochs,
        name="Transformer",
    )

    print(f"\n✓ Transformer Accuracy: {transformer_acc:.2f}%")

    # ========================================================================
    # TEST 2: CNN DISTINGUISHER
    # ========================================================================
    print("\n" + "=" * 60)
    print("TEST 2: CNN DISTINGUISHER")
    print("=" * 60)

    model_cnn = CNNDistinguisher().to(device)
    cnn_acc = train_distinguisher(
        model_cnn,
        train_loader,
        test_loader,
        device,
        epochs=args.epochs,
        name="CNN",
    )

    print(f"\n✓ CNN Accuracy: {cnn_acc:.2f}%")

    # ========================================================================
    # TEST 3: NEXT-BYTE PREDICTOR
    # ========================================================================
    print("\n" + "=" * 60)
    print("TEST 3: NEXT-BYTE PREDICTOR (Cryptanalytic)")
    print("=" * 60)
    print(f"Context size: {args.context_size} bytes")

    # Use combined data for better statistics
    combined_data = np.concatenate([agle_data, chacha_data, urandom_data])
    x_pred, y_pred = build_next_byte_dataset(combined_data, args.context_size)

    split = int(len(x_pred) * 0.8)
    x_train_pred = torch.tensor(x_pred[:split])
    y_train_pred = torch.tensor(y_pred[:split])
    x_test_pred = torch.tensor(x_pred[split:])
    y_test_pred = torch.tensor(y_pred[split:])

    train_loader_pred = DataLoader(
        TensorDataset(x_train_pred, y_train_pred),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    test_loader_pred = DataLoader(
        TensorDataset(x_test_pred, y_test_pred),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    print(f"Train samples: {len(x_train_pred):,}")
    print(f"Test samples: {len(x_test_pred):,}")

    model_predictor = NextBytePredictor(args.context_size).to(device)
    predictor_acc = train_next_byte_predictor(
        model_predictor,
        train_loader_pred,
        test_loader_pred,
        device,
        epochs=args.epochs,
    )

    baseline_predictor = 100.0 / 256.0  # Random guess
    print(f"\n✓ Next-Byte Predictor Accuracy: {predictor_acc:.4f}%")
    print(f"  Baseline (random): {baseline_predictor:.4f}%")
    print(f"  Advantage: {predictor_acc - baseline_predictor:.4f}%")

    # ========================================================================
    # TEST 4: SPECTRAL ENTROPY
    # ========================================================================
    print("\n" + "=" * 60)
    print("TEST 4: SPECTRAL ENTROPY")
    print("=" * 60)

    entropies = test_spectral_entropy(agle_data, chacha_data, urandom_data, args.window)
    ideal_entropy = 1.0

    print(f"\n  AGLE:       {entropies['agle']:.4f} (ideal: {ideal_entropy:.4f})")
    print(f"  ChaCha20:   {entropies['chacha']:.4f} (ideal: {ideal_entropy:.4f})")
    print(f"  /dev/urandom: {entropies['urandom']:.4f} (ideal: {ideal_entropy:.4f})")

    # ========================================================================
    # TEST 5: MUTUAL INFORMATION
    # ========================================================================
    print("\n" + "=" * 60)
    print("TEST 5: MUTUAL INFORMATION (lag 1-10)")
    print("=" * 60)

    mi_agle = test_mutual_information(agle_data)
    mi_chacha = test_mutual_information(chacha_data)
    mi_urandom = test_mutual_information(urandom_data)

    print(f"\n  AGLE:       {mi_agle:.6f} (ideal: ~0.0)")
    print(f"  ChaCha20:   {mi_chacha:.6f} (ideal: ~0.0)")
    print(f"  /dev/urandom: {mi_urandom:.6f} (ideal: ~0.0)")

    # ========================================================================
    # FINAL REPORT
    # ========================================================================
    print("\n" + "=" * 60)
    print("FINAL SECURITY ASSESSMENT")
    print("=" * 60)

    baseline_3class = 100.0 / 3.0

    print(f"\n{'Test':<25} {'AGLE':<12} {'Expected':<12} {'Status':<12}")
    print("-" * 65)

    # Transformer
    status_tf = "✅" if abs(transformer_acc - baseline_3class) < baseline_3class * 0.15 else "⚠️ /❌"
    print(
        f"{'Transformer':<25} {transformer_acc:>10.2f}% {baseline_3class:>10.2f}% {status_tf:>12}"
    )

    # CNN
    status_cnn = "✅" if abs(cnn_acc - baseline_3class) < baseline_3class * 0.15 else "⚠️ /❌"
    print(f"{'CNN':<25} {cnn_acc:>10.2f}% {baseline_3class:>10.2f}% {status_cnn:>12}")

    # Next-Byte Predictor
    status_pred = (
        "✅"
        if predictor_acc - baseline_predictor < 0.5
        else "⚠️" if predictor_acc - baseline_predictor < 1.0 else "❌"
    )
    print(
        f"{'Next-Byte Predictor':<25} {predictor_acc:>10.4f}% {baseline_predictor:>10.4f}% {status_pred:>12}"
    )

    # Spectral Entropy
    entropy_avg = np.mean([entropies["agle"], entropies["chacha"], entropies["urandom"]])
    status_entropy = "✅" if entropy_avg > 0.95 else "⚠️" if entropy_avg > 0.90 else "❌"
    print(f"{'Spectral Entropy':<25} {entropy_avg:>10.4f} {ideal_entropy:>10.4f} {status_entropy:>12}")

    # Mutual Information
    mi_avg = np.mean([mi_agle, mi_chacha, mi_urandom])
    status_mi = "✅" if mi_avg < 0.01 else "⚠️" if mi_avg < 0.05 else "❌"
    print(f"{'Mutual Information':<25} {mi_avg:>10.6f} {0.0:>10.6f} {status_mi:>12}")

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)

    all_good = (
        abs(transformer_acc - baseline_3class) < baseline_3class * 0.15
        and abs(cnn_acc - baseline_3class) < baseline_3class * 0.15
        and predictor_acc - baseline_predictor < 0.5
        and entropy_avg > 0.95
        and mi_avg < 0.01
    )

    if all_good:
        print("\n✅ EXCELLENT: AGLE is cryptographically indistinguishable")
        print("   from ChaCha20 and /dev/urandom under these tests.")
    elif (
        abs(transformer_acc - baseline_3class) < baseline_3class * 0.2
        and abs(cnn_acc - baseline_3class) < baseline_3class * 0.2
    ):
        print("\n✅ GOOD: No significant distinguishing attacks found.")
    else:
        print("\n⚠️ WARNING: Some distinguishability detected.")
        print("   Consider reviewing AGLE design.")

    print()


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"Total runtime: {elapsed:.1f}s ({elapsed/60:.1f}m)")
