#!/usr/bin/env python3

import numpy as np
from math import log, factorial
import sys

print("=== Permutation Entropy Test ===\n")


def load_data(max_uint32=200000):
    if not sys.stdin.isatty():
        raw = sys.stdin.buffer.read()
        usable = (len(raw) // 4) * 4
        if usable == 0:
            return np.array([], dtype=np.uint32)
        return np.frombuffer(raw[:usable], dtype=np.uint32)

    return np.fromfile(
        "dados/bins/agle_bits_1gb.bin",
        dtype=np.uint32,
        count=max_uint32,
    )


def permutation_entropy(series, order=5):
    patterns = {}

    for index in range(len(series) - order):
        window = series[index:index + order]
        key = tuple(np.argsort(window))
        patterns[key] = patterns.get(key, 0) + 1

    counts = np.array(list(patterns.values()), dtype=float)
    p = counts / np.sum(counts)

    entropy = -np.sum(p * np.log(p))
    normalized_entropy = entropy / log(factorial(order))

    return entropy, normalized_entropy, patterns


def main():
    data = load_data(max_uint32=200000)

    if len(data) < 10:
        print("❌ Dados insuficientes para o teste")
        sys.exit(1)

    print("Samples:", len(data))

    for order in [3, 4, 5]:
        entropy, normalized_entropy, patterns = permutation_entropy(data, order)

        print("\nOrder:", order)
        print("Unique patterns:", len(patterns), "/", factorial(order))
        print("Entropy:", entropy)
        print("Normalized entropy:", normalized_entropy)

        if normalized_entropy > 0.98:
            verdict = "EXCELLENT"
        elif normalized_entropy > 0.95:
            verdict = "GOOD"
        elif normalized_entropy > 0.90:
            verdict = "STRUCTURE DETECTED"
        else:
            verdict = "DETERMINISTIC"

        print("Verdict:", verdict)


if __name__ == "__main__":
    main()
