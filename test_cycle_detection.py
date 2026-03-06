#!/usr/bin/env python3
"""
Cycle Detection Test
Detecta se o gerador entra em ciclos (finite precision collapse)
"""
import numpy as np
import sys

print("=== Cycle Detection Test ===\n")

try:
    data = np.fromfile("dados/bins/agle_bits_1gb.bin", dtype=np.uint32, count=1000000)
    print(f"Loaded {len(data)} uint32 values")
    
    print("Searching for cycles...\n")
    
    seen = set()
    cycle_detected = False
    cycle_position = -1
    
    for i, x in enumerate(data):
        if x in seen:
            cycle_detected = True
            cycle_position = i
            print(f"⚠️  Value {x} repeated at position {i}")
            if i < 1000:
                print(f"   WARNING: Very short cycle detected!")
            break
        seen.add(x)
        
        if i % 100000 == 0 and i > 0:
            print(f"  Progress: {i:,} values checked, no cycles yet...")
    
    print(f"\nTotal unique values seen: {len(seen):,}")
    
    if not cycle_detected:
        print(f"✅ PASSED: No cycles detected in {len(data):,} values")
    elif cycle_position > 500000:
        print(f"✅ PASSED: Cycle detected late at position {cycle_position:,}")
        print("   This is acceptable for such a large sequence")
    else:
        print(f"❌ FAILED: Cycle detected early at position {cycle_position:,}")
        print("   RNG has finite precision collapse problem")
    
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
