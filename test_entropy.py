#!/usr/bin/env python3
"""
Shannon Entropy Test
Mede a entropia da sequência - deve estar próximo de 8.0 bits/byte
"""
import numpy as np
from math import log2
import sys

print("=== Shannon Entropy Test ===\n")

try:
    data = np.fromfile("dados/bins/agle_bits_1gb.bin", dtype=np.uint8, count=10000000)
    print(f"Loaded {len(data)} bytes")
    
    values, counts = np.unique(data, return_counts=True)
    
    p = counts / len(data)
    H = -np.sum(p * np.log2(p))
    
    print(f"\nShannon Entropy: {H:.6f} bits/byte")
    print(f"Maximum possible: 8.000000 bits/byte")
    print(f"Difference: {8.0 - H:.6f}\n")
    
    if H >= 7.999:
        print("✅ PASSED: Excellent entropy (7.999 - 8.000)")
    elif H >= 7.99:
        print("✅ PASSED: Very good entropy (7.99 - 7.999)")
    elif H >= 7.95:
        print("⚠️  WARNING: Good entropy but not ideal (7.95 - 7.99)")
    else:
        print(f"❌ FAILED: Low entropy ({H:.3f}) - not suitable for cryptography")
    
    # Show distribution stats
    print(f"\nDistribution stats:")
    print(f"  Unique values: {len(values)}/256")
    print(f"  Min count: {counts.min()}")
    print(f"  Max count: {counts.max()}")
    print(f"  Mean count: {counts.mean():.2f}")
    print(f"  Std dev: {counts.std():.2f}")
    
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
