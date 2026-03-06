#!/usr/bin/env python3
"""
Machine Learning Predictability Test (Correct Version)
Testa previsibilidade REAL do RNG usando classificação de 256 valores
Métrica anterior (±20) estava incorreta - dava ~16% para dados aleatórios
"""
import numpy as np
import sys

print("=== ML Predictability Test (Correct - Classification) ===\n")

try:
    # Check if torch is available
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("⚠️  PyTorch not installed. Skipping ML test.")
        print("Install with: pip install torch")
        sys.exit(0)
    
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
    
    # Limit to 200k bytes for reasonable training time
    if len(data) > 200000:
        data = data[:200000]
    
    window = 16  # Use 16 previous bytes to predict next byte
    
    X = []
    Y = []
    
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        Y.append(data[i+window])
    
    # Normalize inputs to [0,1] range
    X = torch.tensor(X, dtype=torch.float32) / 255.0
    # Targets are class labels (0-255)
    Y = torch.tensor(Y, dtype=torch.long)
    
    # Train/test split to avoid overfitting
    split = int(len(X) * 0.8)
    X_train, Y_train = X[:split], Y[:split]
    X_test, Y_test = X[split:], Y[split:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Window size: {window} bytes")
    print(f"Output classes: 256 (0-255)\n")
    
    # Neural network for 256-class classification
    model = nn.Sequential(
        nn.Linear(window, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 256)  # Output logits for 256 classes
    )
    
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("Training neural network...\n")
    
    for epoch in range(15):
        # Training step
        logits = model(X_train)
        loss = loss_fn(logits, Y_train)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if epoch % 3 == 0:
            print(f"  Epoch {epoch:2d}: loss = {loss.item():.4f}")
    
    print("\nEvaluating on test set...\n")
    
    # Evaluate on test set (unseen data)
    with torch.no_grad():
        logits_test = model(X_test)
        pred_test = torch.argmax(logits_test, dim=1)
        acc_test = (pred_test == Y_test).float().mean().item() * 100
        
        # Also check train accuracy to detect overfitting
        logits_train = model(X_train)
        pred_train = torch.argmax(logits_train, dim=1)
        acc_train = (pred_train == Y_train).float().mean().item() * 100
    
    baseline = 100.0 / 256.0  # Random guess: 0.390625%
    
    print("=" * 60)
    print(f"RESULTS:")
    print(f"  Training accuracy:   {acc_train:.3f}%")
    print(f"  Test accuracy:       {acc_test:.3f}%")
    print(f"  Random baseline:     {baseline:.3f}%")
    print(f"  Test/Baseline ratio: {acc_test/baseline:.2f}x")
    print("=" * 60)
    
    print("\nINTERPRETATION:")
    if acc_test < baseline * 1.5:
        print(f"  {acc_test:.3f}% < {baseline*1.5:.3f}% → ✅ PASSED: No learnable structure")
        result = "PASSED"
    elif acc_test < baseline * 3.0:
        print(f"  {acc_test:.3f}% < {baseline*3.0:.3f}% → ⚠️  WARNING: Some patterns detected")
        result = "WARNING"
    else:
        print(f"  {acc_test:.3f}% ≥ {baseline*3.0:.3f}% → ❌ FAILED: RNG is learnable!")
        result = "FAILED"
    
    # Check for overfitting
    if acc_train > acc_test * 1.5:
        print(f"\n  ⚠️  Overfitting detected (train {acc_train:.2f}% >> test {acc_test:.2f}%)")
    
    print("\nREFERENCE THRESHOLDS:")
    print(f"  0.39% - 0.60%: Truly random")
    print(f"  0.60% - 1.20%: Acceptable")
    print(f"  1.20% - 2.00%: Suspicious")
    print(f"  > 2.00%:       Predictable RNG")
    
    print(f"\nFINAL VERDICT: {result}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
