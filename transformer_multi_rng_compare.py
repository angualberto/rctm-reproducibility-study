#!/usr/bin/env python3
"""
Transformer Distinguishing Attack: Compara múltiplos RNGs contra AGLE v4
- AGLE v4 (reference)
- ChaCha20
- AES-CTR  
- MT19937
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os
import time
import subprocess
import struct

BLOCK_SIZE = 64
BATCH_SIZE = 1024
EPOCHS = 30
LEARNING_RATE = 0.0005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_HEADS = 8
NUM_LAYERS = 4
D_MODEL = 256

print("="*70)
print(f"Transformer Multi-RNG Comparison (CUDA)")
print("="*70)
print(f"Dispositivo: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memória GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print()

# ============= GERAR RNGs =============
print("PASSO 1: Preparando dados de múltiplos RNGs...")
print()

RNG_GENERATORS = {
    "agle_v4": {
        "file": "/home/andre/Imagens/mestrado/Hardware-Induced-Irreversibility-in-Chaotic-Maps-main/versoes/v2/codigo/agle_v3_200mb.bin",
        "generator": None  # já existe
    },
    "chacha20": {
        "file": "/tmp/chacha20_compare.bin",
        "generator": lambda: generate_chacha20(200_000_000)
    },
    "aes_ctr": {
        "file": "/tmp/aes_ctr_compare.bin",
        "generator": lambda: generate_aes_ctr(200_000_000)
    },
    "mt19937": {
        "file": "/tmp/mt19937_compare.bin",
        "generator": lambda: generate_mt19937(200_000_000)
    },
}

def generate_chacha20(size):
    """Gera dados ChaCha20"""
    try:
        from Crypto.Cipher import ChaCha20
        from Crypto.Random import get_random_bytes
    except ImportError:
        print("  Instalando pycryptodome...")
        os.system("/home/andre/Imagens/mestrado/Hardware-Induced-Irreversibility-in-Chaotic-Maps-main/.venv/bin/pip install pycryptodome -q")
        from Crypto.Cipher import ChaCha20
        from Crypto.Random import get_random_bytes
    
    key = get_random_bytes(32)
    nonce = get_random_bytes(12)
    cipher = ChaCha20.new(key=key, nonce=nonce)
    
    output_file = "/tmp/chacha20_compare.bin"
    print(f"  Gerando ChaCha20 ({size/1e6:.0f} MB)...")
    with open(output_file, "wb") as f:
        remaining = size
        chunk_size = 10_000_000
        while remaining > 0:
            to_gen = min(chunk_size, remaining)
            f.write(cipher.encrypt(b"\x00" * to_gen))
            remaining -= to_gen
    return output_file

def generate_aes_ctr(size):
    """Gera dados AES-CTR"""
    try:
        from Crypto.Cipher import AES
        from Crypto.Random import get_random_bytes
    except ImportError:
        print("  Instalando pycryptodome...")
        os.system("/home/andre/Imagens/mestrado/Hardware-Induced-Irreversibility-in-Chaotic-Maps-main/.venv/bin/pip install pycryptodome -q")
        from Crypto.Cipher import AES
        from Crypto.Random import get_random_bytes
    
    key = get_random_bytes(32)
    cipher = AES.new(key, AES.MODE_CTR)
    
    output_file = "/tmp/aes_ctr_compare.bin"
    print(f"  Gerando AES-CTR ({size/1e6:.0f} MB)...")
    with open(output_file, "wb") as f:
        remaining = size
        chunk_size = 10_000_000
        while remaining > 0:
            to_gen = min(chunk_size, remaining)
            f.write(cipher.encrypt(b"\x00" * to_gen))
            remaining -= to_gen
    return output_file

def generate_mt19937(size):
    """Gera dados MT19937"""
    output_file = "/tmp/mt19937_compare.bin"
    print(f"  Gerando MT19937 ({size/1e6:.0f} MB)...")
    with open(output_file, "wb") as f:
        remaining = size
        chunk_size = 10_000_000
        while remaining > 0:
            to_gen = min(chunk_size, remaining)
            data = np.random.MT19937(seed=int(time.time() * 1e6) % (2**32))
            rng = np.random.Generator(data)
            f.write(rng.bytes(to_gen))
            remaining -= to_gen
    return output_file

# Gerar/carregar dados
for rng_name, rng_info in RNG_GENERATORS.items():
    file_path = rng_info["file"]
    if not os.path.exists(file_path):
        if rng_info["generator"]:
            rng_info["generator"]()
        else:
            print(f"  ERRO: {rng_name} não encontrado em {file_path}")
            exit(1)
    else:
        size = os.path.getsize(file_path) / 1e6
        print(f"  {rng_name}: {size:.1f} MB (existente)")

print()

# ============= COMPARAÇÃO PAIRWISE =============
print("PASSO 2: Transformer Distinguishing Attack (Pairwise Comparisons)...")
print()

def load_blocks(file_path, label, max_samples=50000):
    """Carrega blocos de um arquivo binário"""
    data = np.fromfile(file_path, dtype=np.uint8)
    n = len(data) // BLOCK_SIZE
    if max_samples:
        n = min(n, max_samples)
    data = data[:n * BLOCK_SIZE].reshape(n, BLOCK_SIZE)
    labels = np.full(n, label, dtype=np.long)
    return data, labels

class TransformerEncoder(nn.Module):
    def __init__(self, input_size=BLOCK_SIZE):
        super().__init__()
        self.input_projection = nn.Linear(input_size, D_MODEL)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NUM_HEADS,
            dim_feedforward=1024,
            dropout=0.2,
            batch_first=True,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        self.classification_head = nn.Sequential(
            nn.Linear(D_MODEL, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        # x shape: (batch_size, BLOCK_SIZE)
        # Adicionar dimensão de sequência: (batch_size, 1, BLOCK_SIZE)
        x = x.unsqueeze(1)
        # Projetar para d_model: (batch_size, 1, D_MODEL)
        x = self.input_projection(x)
        # Transformador: (batch_size, 1, D_MODEL)
        x = self.transformer_encoder(x)
        # Pegar primeira posição da sequência: (batch_size, D_MODEL)
        x = x[:, 0, :]
        # Classificação
        x = self.classification_head(x)
        return x

def train_and_evaluate(rng1_name, rng2_name):
    """Treina modelo para distinguir dois RNGs"""
    print(f"  [{rng1_name} vs {rng2_name}]", end=" ", flush=True)
    start_time = time.time()
    
    # Carregar dados
    data1, labels1 = load_blocks(RNG_GENERATORS[rng1_name]["file"], 0)
    data2, labels2 = load_blocks(RNG_GENERATORS[rng2_name]["file"], 1)
    
    X = np.concatenate([data1, data2])
    y = np.concatenate([labels1, labels2])
    
    perm = np.random.permutation(len(X))
    X = X[perm]
    y = y[perm]
    
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    X_train = torch.tensor(X_train, dtype=torch.float32) / 255.0
    X_test = torch.tensor(X_test, dtype=torch.float32) / 255.0
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    
    # Modelo
    model = TransformerEncoder().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Treinar
    model.train()
    for epoch in range(EPOCHS):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        scheduler.step()
    
    # Avaliar
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    elapsed = time.time() - start_time
    
    print(f"Acc={accuracy*100:.2f}%, AUC={auc:.4f}, Time={elapsed:.1f}s")
    
    return {
        "accuracy": accuracy,
        "auc": auc,
        "time": elapsed
    }

# Executar comparações
results = {}
rng_list = list(RNG_GENERATORS.keys())

for i, rng1 in enumerate(rng_list):
    for rng2 in rng_list[i+1:]:
        key = f"{rng1}_vs_{rng2}"
        results[key] = train_and_evaluate(rng1, rng2)

# ============= RESULTADOS =============
print()
print("="*70)
print("RESULTADOS FINAIS")
print("="*70)
print()

print("Tabela Comparativa (Acurácia do Distinguidor):")
print()
print(f"{'Comparação':<30} {'Acurácia':<12} {'AUC-ROC':<12} {'Tempo (s)'}")
print("-" * 70)

for key, result in sorted(results.items()):
    rng1, rng2 = key.replace("_vs_", "|").split("|")
    print(f"{rng1} vs {rng2:<20} {result['accuracy']*100:>6.2f}%     {result['auc']:>8.4f}      {result['time']:>6.1f}")

print()
print("Interpretação:")
print("  Acurácia ~50% = RNGs indistinguíveis (criptograficamente seguros)")
print("  Acurácia >70% = RNG distinguível (qualidade reduzida)")
print()
