#!/usr/bin/env python3
"""
N-gram Predictability Test (NumPy Only)
Testa previsibilidade usando análise de n-gramas
Alternativa ao ML quando PyTorch não está disponível
"""
import numpy as np
import sys

print("=== N-gram Predictability Test ===\n")

try:
    # Read data from stdin if available, otherwise from file
    if not sys.stdin.isatty():
        data = np.frombuffer(sys.stdin.buffer.read(), dtype=np.uint8)
        print(f"Read {len(data)} bytes from stdin")
    else:
        try:
            data = np.fromfile("dados/bins/agle_bits_1gb.bin", dtype=np.uint8, count=200000)
            print(f"Read {len(data)} bytes from file")
        except FileNotFoundError:
            print("❌ Error: No stdin and file not found")
            sys.exit(1)
    
    # Limit to 200k bytes
    if len(data) > 200000:
        data = data[:200000]
    
    # Test multiple n-gram sizes
    for n in [2, 3, 4]:
        print(f"\n{'='*60}")
        print(f"Testing {n}-gram model (window = {n-1} bytes)")
        print('='*60)
        
        # Build frequency table: context -> next_byte
        freq_table = {}
        
        for i in range(len(data) - n):
            context = tuple(data[i:i+n-1])
            next_byte = data[i+n-1]
            
            if context not in freq_table:
                freq_table[context] = np.zeros(256, dtype=int)
            freq_table[context][next_byte] += 1
        
        # Train/test split
        split = int(len(data) * 0.8)
        
        # Make predictions on test set
        correct = 0
        total = 0
        
        for i in range(split, len(data) - n):
            context = tuple(data[i:i+n-1])
            actual = data[i+n-1]
            
            if context in freq_table:
                # Predict most frequent byte for this context
                predicted = np.argmax(freq_table[context])
                if predicted == actual:
                    correct += 1
            total += 1
        
        accuracy = (correct / total * 100) if total > 0 else 0
        baseline = 100.0 / 256.0  # 0.390625%
        
        print(f"  Unique contexts: {len(freq_table):,}")
        print(f"  Test predictions: {total:,}")
        print(f"  Correct: {correct}")
        print(f"  Accuracy: {accuracy:.3f}%")
        print(f"  Baseline (random): {baseline:.3f}%")
        print(f"  Ratio: {accuracy/baseline:.2f}x")
        
        if accuracy < baseline * 1.5:
            print(f"  → ✅ PASSED: No learnable patterns")
        elif accuracy < baseline * 3.0:
            print(f"  → ⚠️  WARNING: Some patterns detected")
        else:
            print(f"  → ❌ FAILED: High predictability")
    
    print("\n" + "="*60)
    print("INTERPRETATION:")
    print("  < 0.60%: Truly random")
    print("  0.60% - 1.20%: Acceptable")
    print("  1.20% - 2.00%: Suspicious")
    print("  > 2.00%: Predictable RNG")
    print("="*60)
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
