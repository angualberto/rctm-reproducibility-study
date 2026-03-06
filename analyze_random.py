#!/usr/bin/env python3
"""
Análise de Entropia e Qualidade do AGLE Random Generator
"""

import sys
import numpy as np
from collections import Counter

def shannon_entropy(data):
    """Calcula entropia de Shannon em bits por byte"""
    if len(data) == 0:
        return 0
    # Converte bytes para array de inteiros se necessário
    if isinstance(data, bytes):
        data = bytearray(data)
    counter = Counter(data)
    entropy = 0
    for count in counter.values():
        p = count / len(data)
        entropy -= p * np.log2(p)
    return entropy

def chi_square_test(data):
    """Teste Chi-quadrado para uniformidade"""
    counter = Counter(data)
    expected = len(data) / 256
    chi2 = sum((count - expected)**2 / expected for count in counter.values())
    # Para 255 graus de liberdade, valor crítico ~ 293.25 (p=0.05)
    return chi2

def serial_correlation(data, lag=1):
    """Correlação serial com defasagem"""
    if len(data) < lag + 1:
        return 0
    x = np.array(data[:-lag], dtype=float)
    y = np.array(data[lag:], dtype=float)
    
    if np.std(x) == 0 or np.std(y) == 0:
        return 0
    
    return np.corrcoef(x, y)[0, 1]

def analyze_random_data(filename, max_bytes=None):
    """Análise completa de dados aleatórios"""
    print(f"📊 Analisando: {filename}\n")
    
    with open(filename, 'rb') as f:
        data = f.read(max_bytes) if max_bytes else f.read()
    
    if len(data) == 0:
        print("❌ Arquivo vazio!")
        return
    
    print(f"📦 Tamanho: {len(data):,} bytes ({len(data)/1024:.2f} KB)\n")
    
    # 1. Entropia de Shannon
    entropy = shannon_entropy(data)
    print(f"🔢 Entropia de Shannon: {entropy:.6f} bits/byte")
    print(f"   Ideal: 8.0 bits/byte (100% aleatoriedade)")
    print(f"   Score: {(entropy/8)*100:.2f}%\n")
    
    # 2. Teste Chi-quadrado
    chi2 = chi_square_test(data)
    print(f"📈 Teste Chi-Quadrado: {chi2:.2f}")
    print(f"   Valor crítico (p=0.05): 293.25")
    print(f"   Status: {'✅ PASSA' if chi2 < 293.25 else '❌ FALHA'}\n")
    
    # 3. Distribuição de bytes
    counter = Counter(data)
    min_count = min(counter.values()) if counter else 0
    max_count = max(counter.values()) if counter else 0
    mean_count = len(data) / 256
    print(f"📊 Distribuição de Bytes:")
    print(f"   Mínimo: {min_count} ({min_count/mean_count*100:.1f}% da média)")
    print(f"   Máximo: {max_count} ({max_count/mean_count*100:.1f}% da média)")
    print(f"   Média esperada: {mean_count:.1f}\n")
    
    # 4. Correlação serial
    corr1 = serial_correlation(data, lag=1)
    corr8 = serial_correlation(data, lag=8)
    print(f"🔗 Correlação Serial:")
    print(f"   Lag 1: {corr1:.6f}")
    print(f"   Lag 8: {corr8:.6f}")
    print(f"   Status: {'✅ BOM' if abs(corr1) < 0.1 and abs(corr8) < 0.1 else '⚠️  ATENÇÃO'}\n")
    
    # 5. Análise de runs (sequências)
    runs = 0
    for i in range(len(data) - 1):
        if data[i] != data[i+1]:
            runs += 1
    expected_runs = len(data) - 1
    print(f"🏃 Análise de Runs:")
    print(f"   Runs observados: {runs}")
    print(f"   Runs esperados: ~{expected_runs}")
    print(f"   Razão: {runs/expected_runs:.4f}\n")
    
    # 6. Resumo final
    print("=" * 50)
    scores = []
    scores.append(entropy/8)
    scores.append(1.0 if chi2 < 293.25 else 0.5)
    scores.append(1.0 if abs(corr1) < 0.1 else 0.5)
    
    avg_score = np.mean(scores) * 100
    print(f"🎯 Score Geral: {avg_score:.1f}%")
    
    if avg_score >= 90:
        print("✅ EXCELENTE qualidade de aleatoriedade!")
    elif avg_score >= 70:
        print("✅ BOA qualidade de aleatoriedade")
    elif avg_score >= 50:
        print("⚠️  Qualidade MODERADA")
    else:
        print("❌ Qualidade BAIXA")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Uso: python3 analyze_random.py <arquivo.bin> [max_bytes]")
        sys.exit(1)
    
    filename = sys.argv[1]
    max_bytes = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    analyze_random_data(filename, max_bytes)
