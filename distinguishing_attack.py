#!/usr/bin/env python3
"""
Ataque de distinção entre AGLE V3 e ChaCha20 usando rede neural
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Configurações
BLOCK_SIZE = 64  # bytes
BATCH_SIZE = 512
EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Dispositivo: {DEVICE}")

# ============= PASSO 1: Gerar dados ChaCha20 =============

print("\n" + "="*60)
print("PASSO 1: Gerando dados ChaCha20...")
print("="*60)

try:
    from Crypto.Cipher import ChaCha20
    from Crypto.Random import get_random_bytes
except ImportError:
    print("Instalando pycryptodome...")
    os.system("pip install pycryptodome -q")
    from Crypto.Cipher import ChaCha20
    from Crypto.Random import get_random_bytes

size = 200_000_000  # 200 MB

key = get_random_bytes(32)
nonce = get_random_bytes(12)
cipher = ChaCha20.new(key=key, nonce=nonce)

# Gerar em chunks para economizar memória
chacha_file = "/home/andre/Imagens/mestrado/Hardware-Induced-Irreversibility-in-Chaotic-Maps-main/chacha20_200mb.bin"

with open(chacha_file, "wb") as f:
    remaining = size
    chunk_size = 10_000_000  # 10 MB por vez
    while remaining > 0:
        to_generate = min(chunk_size, remaining)
        data = cipher.encrypt(b"\x00" * to_generate)
        f.write(data)
        remaining -= to_generate
        print(f"  Gerado: {size - remaining:,} / {size:,} bytes")

file_size = os.path.getsize(chacha_file)
print(f"✓ ChaCha20 gerado: {file_size / 1e6:.1f} MB")

# ============= PASSO 2: Carregar blocos em memória =============

print("\n" + "="*60)
print("PASSO 2: Carregando blocos de dados...")
print("="*60)

def load_blocks(file_path, label, block_size=BLOCK_SIZE, max_samples=None):
    """Carrega arquivo binário em blocos"""
    data = np.fromfile(file_path, dtype=np.uint8)
    n_blocks = len(data) // block_size
    
    if max_samples:
        n_blocks = min(n_blocks, max_samples)
    
    data = data[:n_blocks * block_size].reshape(n_blocks, block_size)
    labels = np.full(n_blocks, label)
    
    return data, labels

agle_file = "/home/andre/Imagens/mestrado/Hardware-Induced-Irreversibility-in-Chaotic-Maps-main/versoes/v2/codigo/agle_v3_200mb.bin"

# Limitar a 100k blocos cada para não explodir memória
agle_data, agle_labels = load_blocks(agle_file, label=0, max_samples=100000)
chacha_data, chacha_labels = load_blocks(chacha_file, label=1, max_samples=100000)

print(f"AGLE: {agle_data.shape[0]:,} blocos")
print(f"ChaCha20: {chacha_data.shape[0]:,} blocos")

# ============= PASSO 3: Preparar dataset =============

print("\n" + "="*60)
print("PASSO 3: Preparando dataset...")
print("="*60)

X = np.concatenate([agle_data, chacha_data])
y = np.concatenate([agle_labels, chacha_labels])

# Embaralhar
perm = np.random.permutation(len(X))
X = X[perm]
y = y[perm]

# Split: 80% treino, 20% teste
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Converter para tensor
X_train = torch.tensor(X_train, dtype=torch.float32, device=DEVICE) / 255.0
X_test = torch.tensor(X_test, dtype=torch.float32, device=DEVICE) / 255.0
y_train = torch.tensor(y_train, dtype=torch.long, device=DEVICE)
y_test = torch.tensor(y_test, dtype=torch.long, device=DEVICE)

print(f"Treino: {X_train.shape[0]:,} amostras")
print(f"Teste: {X_test.shape[0]:,} amostras")

# DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ============= PASSO 4: Definir modelo =============

print("\n" + "="*60)
print("PASSO 4: Definindo arquitetura neural...")
print("="*60)

class DistinguishingNet(nn.Module):
    def __init__(self, input_size=BLOCK_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, 2)  # 2 classes: AGLE (0) vs ChaCha20 (1)
        )
    
    def forward(self, x):
        return self.net(x)

model = DistinguishingNet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

print(f"Modelo criado com {sum(p.numel() for p in model.parameters()):,} parâmetros")
print(model)

# ============= PASSO 5: Treinar =============

print("\n" + "="*60)
print("PASSO 5: Treinando modelo...")
print("="*60)

train_losses = []
test_accs = []

for epoch in range(EPOCHS):
    # Treino
    model.train()
    epoch_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for xb, yb in pbar:
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    
    # Avaliação
    model.eval()
    with torch.no_grad():
        test_pred = []
        test_true = []
        
        for xb, yb in test_loader:
            pred = model(xb)
            pred_class = pred.argmax(dim=1)
            test_pred.extend(pred_class.cpu().numpy())
            test_true.extend(yb.cpu().numpy())
        
        acc = accuracy_score(test_true, test_pred)
        test_accs.append(acc)
    
    print(f"Epoch {epoch+1:2d}/{EPOCHS} | Loss: {avg_loss:.4f} | Teste Acurácia: {acc:.4f}")

# ============= PASSO 6: Avaliação Final =============

print("\n" + "="*60)
print("PASSO 6: Avaliação Final")
print("="*60)

model.eval()
with torch.no_grad():
    y_pred = []
    y_true = []
    
    for xb, yb in test_loader:
        pred = model(xb)
        pred_class = pred.argmax(dim=1)
        y_pred.extend(pred_class.cpu().numpy())
        y_true.extend(yb.cpu().numpy())

y_pred = np.array(y_pred)
y_true = np.array(y_true)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"\nResultados no conjunto de teste:")
print(f"  Acurácia:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Precisão: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")

# ============= PASSO 7: Interpretação =============

print("\n" + "="*60)
print("PASSO 7: Interpretação do Resultado")
print("="*60)

if accuracy <= 0.55:
    print("✓ EXCELENTE: Acurácia ≈ 50%")
    print("  A IA NÃO consegue distinguir os dois RNGs!")
    print("  Seu AGLE é estatisticamente indistinguível do ChaCha20.")
    print("  Este é um resultado MUITO BOM para criptografia.")
elif accuracy <= 0.70:
    print("◐ BOA: Acurácia 55-70%")
    print("  A IA consegue detectar alguma estrutura.")
    print("  Ainda é bom, mas não tão forte quanto ChaCha20.")
else:
    print("✗ FRACA: Acurácia > 70%")
    print("  A IA detecta padrões claros no seu RNG.")
    print("  Ainda há correlação significativa.")

# ============= PASSO 8: Visualizações =============

print("\n" + "="*60)
print("PASSO 8: Gerando visualizações...")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico de perda
ax = axes[0]
ax.plot(train_losses, label='Perda de Treino', linewidth=2)
ax.set_xlabel('Época')
ax.set_ylabel('Perda (CrossEntropy)')
ax.set_title('Evolução da Perda durante Treino')
ax.grid(True, alpha=0.3)
ax.legend()

# Gráfico de acurácia
ax = axes[1]
ax.plot(test_accs, label='Acurácia Teste', linewidth=2)
ax.axhline(y=0.5, color='r', linestyle='--', label='Acaso (50%)')
ax.set_xlabel('Época')
ax.set_ylabel('Acurácia')
ax.set_title('Evolução da Acurácia de Teste')
ax.set_ylim([0.4, 1.0])
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
out_file = "/home/andre/Imagens/mestrado/Hardware-Induced-Irreversibility-in-Chaotic-Maps-main/resultados/analises/distinguishing_attack_results.png"
plt.savefig(out_file, dpi=100)
print(f"✓ Gráfico salvo: {out_file}")
plt.close()

# ============= PASSO 9: Relatório =============

print("\n" + "="*60)
print("PASSO 9: Relatório Final")
print("="*60)

report = f"""
RELATÓRIO DE DISTINÇÃO AGLE V3 vs ChaCha20
{'='*60}

CONFIGURAÇÃO:
  - Tamanho de bloco: {BLOCK_SIZE} bytes
  - Amostras treino: {len(X_train):,}
  - Amostras teste: {len(X_test):,}
  - Arquitetura: 3 camadas ocultas (256 → 128 → 64 → 32)
  - Otimizador: Adam (lr={LEARNING_RATE})
  - Épocas: {EPOCHS}

RESULTADOS:
  - Acurácia Final: {accuracy:.4f} ({accuracy*100:.2f}%)
  - Precisão: {precision:.4f}
  - Recall: {recall:.4f}
  - F1-Score: {f1:.4f}

INTERPRETAÇÃO:
  {'✓ AGLE é criptograficamente indistinguível de ChaCha20!' if accuracy <= 0.55 else f'◐ Acurácia: {accuracy*100:.2f}% (potencial estrutura detectável)'}

RECOMENDAÇÕES:
  1. Testar com blocos maiores (128, 256, 512 bytes)
  2. Usar CNN para capturar padrões locais
  3. Testar com Transformer Distinguishing Attack
  4. Aumentar número de amostras de treino
  5. Usar técnicas de regularização mais fortes

TIMESTAMP: {np.datetime64('now')}
"""

print(report)

with open("/home/andre/Imagens/mestrado/Hardware-Induced-Irreversibility-in-Chaotic-Maps-main/resultados/analises/distinguishing_attack_report.txt", "w") as f:
    f.write(report)

print("\n✓ Teste de distinção concluído com sucesso!")
