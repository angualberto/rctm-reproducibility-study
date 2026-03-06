#!/usr/bin/env python3

import numpy as np
from collections import Counter
from math import log2
import sys

print("=== Symbolic Dynamics Attack ===\n")


def load_uint32(max_count=200000):
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


data = load_uint32(max_count=200000)
if len(data) < 1000:
    print("❌ Dados insuficientes para symbolic dynamics")
    sys.exit(1)

if len(data) > 200000:
    data = data[:200000]

median = np.median(data)
symbols = (data > median).astype(np.uint8)

print("Samples:", len(data))
print("Median threshold:", median)

window = 8
words = []

for i in range(len(symbols) - window):
    words.append(tuple(symbols[i:i + window]))

counts = Counter(words)
total = sum(counts.values())

entropy = 0.0
for count in counts.values():
    probability = count / total
    entropy -= probability * log2(probability)

max_entropy = log2(2 ** window)
normalized = entropy / max_entropy

print("\nUnique symbolic words:", len(counts), "/", 2 ** window)
print("Symbolic entropy:", entropy)
print("Max:", max_entropy)
print("Normalized:", normalized)

if normalized > 0.98:
    print("\n✅ No symbolic structure detected")
else:
    print("\n❌ Chaotic map detected")
