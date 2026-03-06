#!/usr/bin/env python3

import numpy as np
from collections import Counter
from math import log2
import sys

print("=== Distinguishing Attack ===\n")


def load_uint8(max_count=200000):
    if not sys.stdin.isatty():
        data = np.frombuffer(sys.stdin.buffer.read(), dtype=np.uint8)
        return data

    return np.fromfile(
        "dados/bins/agle_bits_1gb.bin",
        dtype=np.uint8,
        count=max_count,
    )


data = load_uint8(max_count=200000)
if len(data) < 1000:
    print("❌ Dados insuficientes para distinguishing attack")
    sys.exit(1)

if len(data) > 200000:
    data = data[:200000]

window = 4
patterns = []

for i in range(len(data) - window):
    patterns.append(tuple(data[i:i + window]))

counts = Counter(patterns)
total = sum(counts.values())

entropy = 0.0
for value in counts.values():
    probability = value / total
    entropy -= probability * log2(probability)

max_entropy = log2(256 ** window)
normalized = entropy / max_entropy

print("Samples:", len(data))
print("Window:", window)
print("Unique patterns:", len(counts))
print("Entropy:", entropy)
print("Max entropy:", max_entropy)
print("Normalized:", normalized)

if normalized > 0.99:
    print("\n✅ Indistinguishable from random")
else:
    print("\n❌ Distinguishable RNG")
