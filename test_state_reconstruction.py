#!/usr/bin/env python3

import numpy as np
import sys

print("=== State Reconstruction Attack ===\n")


def load_uint32(max_count=100000):
    if not sys.stdin.isatty():
        raw = sys.stdin.buffer.read()
        usable = (len(raw) // 4) * 4
        if usable == 0:
            return np.array([], dtype=np.uint32)
        return np.frombuffer(raw[:usable], dtype=np.uint32)

    return np.fromfile(
        "dados/bins/agle_bits_1gb.bin",
        dtype=np.uint32,
        count=max_count,
    )


data = load_uint32(max_count=100000)
if len(data) < 1000:
    print("❌ Dados insuficientes para reconstrução de estado")
    sys.exit(1)

series = data / np.max(data)

embedding = 4
delay = 1

print("Samples:", len(series))
print("Embedding dimension:", embedding)
print("Delay:", delay)

X = []
for i in range(len(series) - (embedding - 1) * delay):
    X.append(series[i:i + embedding * delay:delay])

X = np.array(X, dtype=np.float64)
print("Reconstructed vectors:", len(X))

try:
    from sklearn.neighbors import NearestNeighbors
except ImportError:
    print("❌ scikit-learn não instalado. Instale com: pip install scikit-learn")
    sys.exit(1)

nbrs = NearestNeighbors(n_neighbors=2, algorithm="auto").fit(X)
distances, indices = nbrs.kneighbors(X)

mean_dist = float(np.mean(distances[:, 1]))

print("\nMean neighbor distance:", mean_dist)

if mean_dist < 0.0005:
    print("❌ Possible deterministic structure")
else:
    print("✅ No recoverable state structure")
