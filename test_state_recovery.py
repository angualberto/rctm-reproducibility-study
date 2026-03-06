#!/usr/bin/env python3
"""
State Recovery Attack Test
Detecta repetições de estado que indicam vulnerabilidade
"""
import numpy as np
import sys

print("=== State Recovery Attack Test ===\n")

try:
    data = np.fromfile("dados/bins/agle_bits_1gb.bin", dtype=np.uint32, count=10000000)
    print(f"Loaded {len(data)} uint32 values")
    
    window = 10
    repetitions_found = 0
    
    print(f"Testing with window size: {window}")
    print("Searching for state repetitions...\n")
    
    for i in range(len(data)-window):
        X = data[i:i+window]
        Y = data[i+window]
        
        if np.all(X == X[0]):
            repetitions_found += 1
            if repetitions_found <= 5:  # Show first 5
                print(f"Possible state repetition at index {i}: value={X[0]}")
    
    print(f"\nTotal repetitions found: {repetitions_found}")
    
    if repetitions_found == 0:
        print("✅ PASSED: No state repetitions detected")
    elif repetitions_found < 10:
        print("⚠️  WARNING: Few repetitions (might be random)")
    else:
        print("❌ FAILED: Multiple state repetitions detected - RNG may be predictable")
    
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
