#!/usr/bin/env python3
"""Teste rápido da qualidade com função harmônica adicionada"""

import subprocess
import struct
import numpy as np
from scipy import stats

AGLE_BIN = "./versoes/v2/codigo/agle_v3_experimental"

def get_outputs(seed, n=1000):
    """Gera n valores"""
    cmd = [AGLE_BIN, "--stdout", str(seed)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    data = []
    for _ in range(n):
        chunk = proc.stdout.read(8)
        if not chunk: break
        data.append(struct.unpack('<Q', chunk)[0])
    
    proc.terminate()
    return np.array(data, dtype=np.uint64)

print("="*60)
print("TESTE RÁPIDO - AGLE V3 COM FUNÇÃO HARMÔNICA")
print("="*60)

outputs = get_outputs(123456789, n=5000)
norm = outputs / (2**64)

# Testes básicos
print(f"\n1. ESTATÍSTICAS BÁSICAS:")
print(f"   Média: {norm.mean():.6f} (esperado: 0.5)")
print(f"   Desvio: {norm.std():.6f} (esperado: 0.289)")

# Teste KS
ks_stat, ks_pval = stats.kstest(norm, 'uniform')
print(f"\n2. TESTE KOLMOGOROV-SMIRNOV:")
print(f"   p-valor: {ks_pval:.4f}")
print(f"   Resultado: {'✓ PASSOU' if ks_pval > 0.05 else '✗ FALHOU'}")

# Autocorrelação
if len(norm) > 100:
    acf = np.corrcoef(norm[:-1], norm[1:])[0,1]
    print(f"\n3. AUTOCORRELAÇÃO (LAG-1):")
    print(f"   Valor: {acf:.6f}")
    print(f"   Status: {'✓ BOM' if abs(acf) < 0.05 else '⚠ DETECTADO'}")

# Entropia
hist, _ = np.histogram(norm, bins=256)
entropy = -np.sum((hist/len(norm)) * np.log2((hist+1e-10)/len(norm)))
max_entropy = 8.0
print(f"\n4. ENTROPIA (256 bins):")
print(f"   Entropia: {entropy:.2f} bits/byte (máximo: {max_entropy:.2f})")
print(f"   Taxa: {entropy/max_entropy*100:.1f}%")

# Chi-square
hist, _ = np.histogram(norm, bins=100)
expected = len(norm) / 100
chi2 = np.sum((hist - expected)**2 / expected)
chi2_pval = 1 - stats.chi2.cdf(chi2, df=99)
print(f"\n5. TESTE CHI-SQUARE:")
print(f"   p-valor: {chi2_pval:.4f}")
print(f"   Resultado: {'✓ PASSOU' if chi2_pval > 0.05 else '✗ FALHOU'}")

print("\n" + "="*60)
if ks_pval > 0.05 and chi2_pval > 0.05 and abs(acf) < 0.05:
    print("✓ TODOS OS TESTES PASSARAM COM FUNÇÃO HARMÔNICA!")
else:
    print("⚠ Alguns testes com avisos - analisar resultados")
print("="*60)
