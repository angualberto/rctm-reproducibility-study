#!/usr/bin/env python3
"""
Transformer Distinguishing Attack: AGLE V3 vs ChaCha20
Usa GPU CUDA para treino rápido
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time

# Configurações
BLOCK_SIZE = 64  # bytes
BATCH_SIZE = 1024
EPOCHS = 30
LEARNING_RATE = 0.0005
WARMUP_EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_HEADS = 8
NUM_LAYERS = 4
D_MODEL = 256

print("="*70)
print(f"Transformer Distinguishing Attack com CUDA")
print("="*70)
print(f"Dispositivo: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memória GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print()

# ============= PASSO 1: Gerar ChaCha20 =============

print("PASSO 1: Preparando dados ChaCha20...")

try:
    from Crypto.Cipher import ChaCha20
    from Crypto.Random import get_random_bytes
except ImportError:
    print("Instalando pycryptodome...")
    os.system("/home/andre/Imagens/mestrado/Hardware-Induced-Irreversibility-in-Chaotic-Maps-main/.venv/bin/pip install pycryptodome -q")
    from Crypto.Cipher import ChaCha20
    from Crypto.Random import get_random_bytes

size = 200_000_000
key = get_random_bytes(32)
nonce = get_random_bytes(12)
cipher = ChaCha20.new(key=key, nonce=nonce)

chacha_file = "/home/andre/Imagens/mestrado/Hardware-Induced-Irreversibility-in-Chaotic-Maps-main/chacha20_cuda.bin"

if not os.path.exists(chacha_file):
    print(f"  Gerando ChaCha20 ({size/1e6:.0f} MB)...")
    with open(chacha_file, "wb") as f:
        remaining = size
        chunk_size = 10_000_000
        while remaining > 0:
            to_gen = min(chunk_size, remaining)
            f.write(cipher.encrypt(b"\x00" * to_gen))
            remaining -= to_gen
            print(f"    {size - remaining:,} / {size:,} bytes")
else:
    print(f"  ChaCha20 já existe: {os.path.getsize(chacha_file)/1e6:.0f} MB")

# ============= PASSO 2: Carregar blocos =============

print("\nPASSO 2: Carregando blocos em GPU...")

def load_blocks(file_path, label, block_size=BLOCK_SIZE, max_samples=None):
    data = np.fromfile(file_path, dtype=np.uint8)
    n = len(data) // block_size
    if max_samples:
        n = min(n, max_samples)
    data = data[:n * block_size].reshape(n, block_size)
    labels = np.full(n, label, dtype=np.long)
    return data, labels

agle_file = "/home/andre/Imagens/mestrado/Hardware-Induced-Irreversibility-in-Chaotic-Maps-main/versoes/v2/codigo/agle_v3_200mb.bin"

# 50k amostras cada = 3.2M treino + 0.8M teste
agle_data, agle_labels = load_blocks(agle_file, 0, max_samples=50000)
chacha_data, chacha_labels = load_blocks(chacha_file, 1, max_samples=50000)

print(f"  AGLE: {len(agle_data):,} blocos")
print(f"  ChaCha20: {len(chacha_data):,} blocos")

X = np.concatenate([agle_data, chacha_data])
y = np.concatenate([agle_labels, chacha_labels])

perm = np.random.permutation(len(X))
X = X[perm]
y = y[perm]

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Converter para tensor e transferir para GPU
X_train = torch.tensor(X_train, dtype=torch.float32) / 255.0
X_test = torch.tensor(X_test, dtype=torch.float32) / 255.0
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

print(f"\n  Treino: {X_train.shape}")
print(f"  Teste: {X_test.shape}")

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# ============= PASSO 3: Transformer =============

print("\nPASSO 3: Definindo Transformer...")

class TransformerEncoder(nn.Module):
    def __init__(self, input_size, d_model, num_heads, num_layers, dim_feedforward=1024):
        super().__init__()
        
        self.input_projection = nn.Linear(input_size, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=0.2,
            batch_first=True,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classification_head = nn.Sequential(
            nn.Linear(d_model, 512),
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
        # x: (batch, block_size)
        x = self.input_projection(x)  # (batch, d_model)
        x = x.unsqueeze(1)  # (batch, 1, d_model) - adiciona dimensão de sequência
        
        x = self.transformer_encoder(x)  # (batch, 1, d_model)
        x = x[:, 0, :]  # (batch, d_model) - pega token [CLS]
        
        x = self.classification_head(x)  # (batch, 2)
        return x

model = TransformerEncoder(
    input_size=BLOCK_SIZE,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nParâmetros totais: {total_params:,}")
print(f"Parâmetros treináveis: {trainable_params:,}")

# ============= PASSO 4: Treinar =============

print("\nPASSO 4: Treinando com CUDA...")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=EPOCHS, T_mult=1, eta_min=1e-6)
loss_fn = nn.CrossEntropyLoss()

best_acc = 0.0
train_losses = []
test_accs = []
test_aucs = []

start_time = time.time()

for epoch in range(EPOCHS):
    # ===== TREINO =====
    model.train()
    epoch_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:2d}/{EPOCHS}", leave=False)
    for xb, yb in pbar:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        
        optimizer.zero_grad()
        
        pred = model(xb)
        loss = loss_fn(pred, yb)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    
    # ===== VALIDAÇÃO =====
    model.eval()
    with torch.no_grad():
        y_pred = []
        y_true = []
        y_probs = []
        
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            
            pred = model(xb)
            pred_probs = torch.softmax(pred, dim=1)
            pred_class = pred.argmax(dim=1)
            
            y_pred.extend(pred_class.cpu().numpy())
            y_true.extend(yb.cpu().numpy())
            y_probs.extend(pred_probs[:, 1].cpu().numpy())
        
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        y_probs = np.array(y_probs)
        
        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_probs)
        
        test_accs.append(acc)
        test_aucs.append(auc)
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "/home/andre/Imagens/mestrado/Hardware-Induced-Irreversibility-in-Chaotic-Maps-main/transformer_best.pth")
    
    scheduler.step()
    
    print(f"Epoch {epoch+1:2d}/{EPOCHS} | Loss: {avg_loss:.4f} | Acc: {acc:.4f} | AUC: {auc:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

elapsed = time.time() - start_time
print(f"\nTreino concluído em {elapsed/60:.1f} minutos")

# ============= PASSO 5: Avaliação Final =============

print("\n" + "="*70)
print("PASSO 5: Avaliação Final")
print("="*70)

model.load_state_dict(torch.load("/home/andre/Imagens/mestrado/Hardware-Induced-Irreversibility-in-Chaotic-Maps-main/transformer_best.pth"))
model.eval()

with torch.no_grad():
    y_pred = []
    y_true = []
    y_probs = []
    
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        
        pred = model(xb)
        pred_probs = torch.softmax(pred, dim=1)
        pred_class = pred.argmax(dim=1)
        
        y_pred.extend(pred_class.cpu().numpy())
        y_true.extend(yb.cpu().numpy())
        y_probs.extend(pred_probs[:, 1].cpu().numpy())

y_pred = np.array(y_pred)
y_true = np.array(y_true)
y_probs = np.array(y_probs)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_probs)

print(f"\nResultados finais (Conjunto de Teste):")
print(f"  Acurácia:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Precisão: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")
print(f"  AUC-ROC:   {auc:.4f}")

# ============= PASSO 6: Interpretação =============

print("\n" + "="*70)
print("PASSO 6: Interpretação")
print("="*70)

if accuracy <= 0.55:
    print("✓ EXCELENTE: Acurácia ≈ 50%")
    print("  Seu RNG é INDISTINGUÍVEL de ChaCha20!")
    print("  Resultado criptograficamente FORTE.")
elif accuracy <= 0.70:
    print("◐ BOA: Acurácia 55-70%")
    print("  Alguma estrutura detectável, mas ainda bom.")
else:
    print("✗ FRACA: Acurácia > 70%")
    print("  Padrões claros detectáveis.")

# ============= PASSO 7: Visualizações =============

print("\nPASSO 7: Gerando visualizações...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Perda
ax = axes[0, 0]
ax.semilogy(train_losses, linewidth=2, label='Treino')
ax.set_xlabel('Época')
ax.set_ylabel('Perda (log scale)')
ax.set_title('Evolução da Perda')
ax.grid(True, alpha=0.3)
ax.legend()

# 2. Acurácia
ax = axes[0, 1]
ax.plot(test_accs, linewidth=2, label='Acurácia Teste')
ax.axhline(y=0.5, color='r', linestyle='--', label='Acaso')
ax.set_xlabel('Época')
ax.set_ylabel('Acurácia')
ax.set_title('Evolução da Acurácia')
ax.set_ylim([0.4, 1.0])
ax.grid(True, alpha=0.3)
ax.legend()

# 3. AUC-ROC
ax = axes[1, 0]
ax.plot(test_aucs, linewidth=2, label='AUC-ROC')
ax.set_xlabel('Época')
ax.set_ylabel('AUC-ROC')
ax.set_title('Evolução do AUC-ROC')
ax.set_ylim([0.4, 1.0])
ax.grid(True, alpha=0.3)
ax.legend()

# 4. Matriz de confusão
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
ax = axes[1, 1]
im = ax.imshow(cm, cmap='Blues', aspect='auto')
ax.set_xlabel('Predito')
ax.set_ylabel('Real')
ax.set_title('Matriz de Confusão')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['AGLE', 'ChaCha20'])
ax.set_yticklabels(['AGLE', 'ChaCha20'])

# Adiciona valores nas células
for i in range(2):
    for j in range(2):
        text = ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black")

plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("/home/andre/Imagens/mestrado/Hardware-Induced-Irreversibility-in-Chaotic-Maps-main/resultados/analises/transformer_cuda_results.png", dpi=100)
print("✓ Gráfico salvo")
plt.close()

# ============= RELATÓRIO =============

report = f"""
TRANSFORMER DISTINGUISHING ATTACK - AGLE V3 vs ChaCha20
{'='*70}

HARDWARE:
  GPU: {torch.cuda.get_device_name(0)}
  CUDA: Version {torch.version.cuda}

CONFIGURAÇÃO:
  Block Size: {BLOCK_SIZE} bytes
  D_Model: {D_MODEL}
  Num Heads: {NUM_HEADS}
  Num Layers: {NUM_LAYERS}
  Total Params: {total_params:,}
  Trainable Params: {trainable_params:,}
  
DATASET:
  Treino: {len(X_train):,} amostras
  Teste: {len(X_test):,} amostras
  
TREINO:
  Épocas: {EPOCHS}
  Batch Size: {BATCH_SIZE}
  Learning Rate: {LEARNING_RATE}
  Tempo Total: {elapsed/60:.1f} minutos
  
RESULTADOS:
  Acurácia Final: {accuracy:.4f} ({accuracy*100:.2f}%)
  Precisão: {precision:.4f}
  Recall: {recall:.4f}
  F1-Score: {f1:.4f}
  AUC-ROC: {auc:.4f}
  
MELHOR ACURÁCIA: {max(test_accs):.4f} (Época {np.argmax(test_accs)+1})

CONCLUSÃO:
  {'✓ AGLE é criptograficamente indistinguível!' if accuracy <= 0.55 else f'◐ Accuracy: {accuracy*100:.2f}% (Detectável a este nível)'}

DATA: {np.datetime64('now')}
"""

with open("/home/andre/Imagens/mestrado/Hardware-Induced-Irreversibility-in-Chaotic-Maps-main/resultados/analises/transformer_cuda_report.txt", "w") as f:
    f.write(report)

print("\n" + "="*70)
print(report)
print("="*70)
print("\n✓ Teste concluído com sucesso!")
