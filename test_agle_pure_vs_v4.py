#!/usr/bin/env python3
"""
Comparação: AGLE Pure vs AGLE v4 (com SHAKE256)
Transformer Distinguishing Attack
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score
import os
import time

BLOCK_SIZE = 64
BATCH_SIZE = 1024
EPOCHS = 30
LEARNING_RATE = 0.0005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*70)
print("AGLE PURE vs AGLE v4 - Transformer Distinguishing Attack")
print("="*70)
print(f"Dispositivo: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

class TransformerEncoder(nn.Module):
    def __init__(self, input_size=BLOCK_SIZE):
        super().__init__()
        self.input_projection = nn.Linear(input_size, 256)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.2,
            batch_first=True,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.classification_head = nn.Sequential(
            nn.Linear(256, 512),
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
        x = x.unsqueeze(1)
        x = self.input_projection(x)
        x = self.transformer_encoder(x)
        x = x[:, 0, :]
        x = self.classification_head(x)
        return x

def load_blocks(file_path, label, max_samples=50000):
    data = np.fromfile(file_path, dtype=np.uint8)
    n = len(data) // BLOCK_SIZE
    if max_samples:
        n = min(n, max_samples)
    data = data[:n * BLOCK_SIZE].reshape(n, BLOCK_SIZE)
    labels = np.full(n, label, dtype=np.long)
    return data, labels

def train_and_evaluate(name1, name2, file1, file2):
    print(f"Teste: [{name1} vs {name2}]", end=" ", flush=True)
    start_time = time.time()
    
    # Carregar dados
    data1, labels1 = load_blocks(file1, 0)
    data2, labels2 = load_blocks(file2, 1)
    
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
    
    return accuracy, auc, elapsed

print("Carregando dados...")
print()

agle_v4_file = "/home/andre/Imagens/mestrado/Hardware-Induced-Irreversibility-in-Chaotic-Maps-main/versoes/v2/codigo/agle_v3_200mb.bin"
agle_pure_file = "/home/andre/Imagens/mestrado/Hardware-Induced-Irreversibility-in-Chaotic-Maps-main/versoes/v2/codigo/agle_pure_200mb.bin"

print("COMPARAÇÕES")
print()

print("1. AGLE Pure vs AGLE v4:")
acc1, auc1, time1 = train_and_evaluate("AGLE Pure", "AGLE v4", agle_pure_file, agle_v4_file)

print()
print("="*70)
print("RESULTADO")
print("="*70)
print()

if acc1 < 52:
    status = "✅ INDISTINGUÍVEL"
    conclusion = "Ambos têm qualidade criptográfica equivalente"
else:
    status = "❌ DISTINGUÍVEL"
    conclusion = "AGLE Pure pode ter diferenças detectáveis"

print(f"AGLE Pure vs AGLE v4")
print(f"  Acurácia:   {acc1*100:.2f}% {status}")
print(f"  AUC-ROC:    {auc1:.4f}")
print(f"  Conclusão:  {conclusion}")
print()

if acc1 < 52:
    print("✅ SUCESSO!")
    print("   AGLE Pure mantém a qualidade criptográfica de AGLE v4")
    print("   sem depender de OpenSSL ou SHAKE256")
else:
    print("⚠️  ATENÇÃO")
    print("   AGLE Pure é distinguível de AGLE v4")
    print("   Possível necessidade de ajustes na dinâmica caótica")

print()
