#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn
from Crypto.Cipher import ChaCha20
from torch.utils.data import DataLoader, TensorDataset

print("=== Transformer Distinguishing Attack ===\n")


class RNGTransformer(nn.Module):
    def __init__(self, num_layers=4):
        super().__init__()

        self.embedding = nn.Linear(1, 32)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=32,
            nhead=4,
            batch_first=True,
            dim_feedforward=128,
            dropout=0.1,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.fc = nn.Linear(32, 3)

    def forward(self, value):
        value = value.unsqueeze(-1)
        value = self.embedding(value)
        value = self.transformer(value)
        value = value.mean(dim=1)
        return self.fc(value)


def generate_agle_bytes(samples, window, agle_bin, seed):
    total_bytes = samples * window
    print("Generating AGLE...")

    process = subprocess.Popen(
        [agle_bin, "--stdout", str(seed)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    data = process.stdout.read(total_bytes)
    process.terminate()

    if len(data) < total_bytes:
        raise RuntimeError(
            f"AGLE output insuficiente: esperado {total_bytes}, obtido {len(data)}"
        )

    return np.frombuffer(data, dtype=np.uint8)


def generate_chacha_bytes(samples, window):
    total_bytes = samples * window
    print("Generating ChaCha20...")

    key = os.urandom(32)
    nonce = os.urandom(8)

    cipher = ChaCha20.new(key=key, nonce=nonce)
    return np.frombuffer(cipher.encrypt(b"\x00" * total_bytes), dtype=np.uint8)


def generate_urandom_bytes(samples, window):
    total_bytes = samples * window
    print("Reading /dev/urandom...")

    with open("/dev/urandom", "rb") as file:
        return np.frombuffer(file.read(total_bytes), dtype=np.uint8)


def build_windows(data, label, window):
    usable = (len(data) // window) * window
    data = data[:usable]
    shaped = data.reshape(-1, window)

    labels = np.full((shaped.shape[0],), label, dtype=np.int64)
    return shaped, labels


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    use_cuda = device.type == "cuda"

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=use_cuda)
            batch_y = batch_y.to(device, non_blocking=use_cuda)

            logits = model(batch_x)
            pred = torch.argmax(logits, dim=1)

            correct += (pred == batch_y).sum().item()
            total += batch_y.size(0)

    return (correct / total * 100.0) if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=123456789)
    parser.add_argument("--agle-bin", default="./versoes/v2/binarios/agle_v2_improved")
    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)

    if not os.path.exists(args.agle_bin):
        print(f"❌ Binário AGLE não encontrado: {args.agle_bin}")
        sys.exit(1)

    window = args.window
    samples = args.samples

    agle_data = generate_agle_bytes(samples, window, args.agle_bin, args.seed)
    chacha_data = generate_chacha_bytes(samples, window)
    urandom_data = generate_urandom_bytes(samples, window)

    x1, y1 = build_windows(agle_data, 0, window)
    x2, y2 = build_windows(chacha_data, 1, window)
    x3, y3 = build_windows(urandom_data, 2, window)

    x = np.vstack([x1, x2, x3]).astype(np.float32) / 255.0
    y = np.concatenate([y1, y2, y3])

    permutation = np.random.permutation(len(x))
    x = x[permutation]
    y = y[permutation]

    split = int(len(x) * 0.8)

    x_train = torch.tensor(x[:split], dtype=torch.float32)
    y_train = torch.tensor(y[:split], dtype=torch.long)
    x_test = torch.tensor(x[split:], dtype=torch.float32)
    y_test = torch.tensor(y[split:], dtype=torch.long)

    if args.device == "cuda":
        if not torch.cuda.is_available():
            print("❌ CUDA solicitada, mas não está disponível neste ambiente")
            sys.exit(1)
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_cuda = device.type == "cuda"

    if use_cuda:
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=use_cuda,
        persistent_workers=args.workers > 0,
    )
    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=use_cuda,
        persistent_workers=args.workers > 0,
    )

    print(f"Train samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    print(f"Window: {window} bytes")
    print(f"Transformer layers: {args.layers}")
    print(f"Classes: 3 (0=AGLE, 1=ChaCha20, 2=/dev/urandom)")

    if use_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Device: {device} ({gpu_name})")
    else:
        print(f"Device: {device}")
    print("")

    model = RNGTransformer(num_layers=args.layers).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    print("Training Transformer...\n")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        total_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=use_cuda)
            batch_y = batch_y.to(device, non_blocking=use_cuda)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=use_cuda, dtype=torch.float16):
                logits = model(batch_x)
                loss = loss_fn(logits, batch_y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            total_batches += 1

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            avg_loss = running_loss / max(total_batches, 1)
            train_acc = evaluate(model, train_loader, device)
            print(f"epoch {epoch:2d} loss {avg_loss:.6f} train_acc {train_acc:.2f}%")

    print("\nEvaluating...")
    acc = evaluate(model, test_loader, device)

    print("\n==============================")
    print("RESULT")
    print("==============================")
    print(f"Accuracy: {acc:.4f}%")

    baseline = 100.0 / 3.0
    print(f"Random baseline: {baseline:.4f}%")

    if acc < baseline * 1.15:
        print("\n✅ PASS: AGLE indistinguishable from ChaCha20 and /dev/urandom")
    elif acc < baseline * 1.5:
        print("\n⚠️ Slight distinguishability")
    else:
        print("\n❌ RNG distinguishable")


if __name__ == "__main__":
    main()
