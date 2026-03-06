#!/usr/bin/env python3
"""
Testes Compreensivos: AGLE Pure (Avalanche Encadeado)
1. N-gram Predictability
2. Cycle Detection
3. Linear Correlation
4. Spectral Entropy
"""

import numpy as np
from collections import Counter, defaultdict
import struct
import time

print("="*70)
print("TESTES COMPREENSIVOS: AGLE PURE (AVALANCHE ENCADEADO)")
print("="*70)
print()

# Arquivo de dados
data_file = "/home/andre/Imagens/mestrado/Hardware-Induced-Irreversibility-in-Chaotic-Maps-main/versoes/v2/codigo/agle_pure_cascaded_200mb.bin"

# ============================================================================
# TESTE 1: N-GRAM PREDICTABILITY
# ============================================================================

print("TESTE 1: N-GRAM PREDICTABILITY")
print("-" * 70)
print()

def test_ngram_predictability(file_path, max_bytes=200000):
    """Testa previsibilidade usando n-gramas de 1 a 4 bytes"""
    
    data = np.fromfile(file_path, dtype=np.uint8, count=max_bytes)
    
    results = {}
    
    for n in [1, 2, 3, 4]:
        print(f"  Testando {n}-gram...", end=" ", flush=True)
        
        # Construir n-gramas
        ngrams = []
        for i in range(len(data) - n):
            ngram = tuple(data[i:i+n])
            ngrams.append(ngram)
        
        # Contar frequências
        counts = Counter(ngrams)
        
        # Calcular entropia
        total = len(ngrams)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        
        # Entropia máxima para n bytes
        max_entropy = 8 * n
        
        # Normalizar
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Calcular previsibilidade (inverso da entropia normalizada)
        predictability = 1.0 - normalized_entropy
        
        # Distribuição uniforme esperada
        unique_ngrams = len(counts)
        expected_unique = min(256**n, total)
        coverage = unique_ngrams / expected_unique
        
        results[n] = {
            'entropy': entropy,
            'max_entropy': max_entropy,
            'normalized': normalized_entropy,
            'predictability': predictability,
            'unique': unique_ngrams,
            'coverage': coverage
        }
        
        print(f"Entropia={entropy:.2f}/{max_entropy:.0f} ({normalized_entropy*100:.2f}%), Prev={predictability*100:.4f}%")
    
    return results

ngram_results = test_ngram_predictability(data_file)

print()
print("Análise N-gram:")
for n, res in ngram_results.items():
    status = "✅ EXCELENTE" if res['predictability'] < 0.01 else "⚠️ DETECTÁVEL" if res['predictability'] < 0.05 else "❌ ALTO"
    print(f"  {n}-gram: Previsibilidade {res['predictability']*100:.4f}% {status}")

print()

# ============================================================================
# TESTE 2: CYCLE DETECTION
# ============================================================================

print("TESTE 2: CYCLE DETECTION")
print("-" * 70)
print()

def test_cycle_detection(file_path, max_values=10_000_000, window_size=100):
    """Detecta ciclos procurando sequências repetidas"""
    
    print(f"  Carregando {max_values} valores uint32...", end=" ", flush=True)
    data = np.fromfile(file_path, dtype=np.uint32, count=max_values)
    print(f"{len(data)} carregados")
    
    print(f"  Procurando ciclos (janela = {window_size} valores)...", end=" ", flush=True)
    
    start = time.time()
    
    # Procurar por sequências repetidas de tamanho window_size
    for i in range(len(data) - 2 * window_size):
        window1 = data[i:i+window_size]
        
        # Procurar correspondência nos próximos valores
        for j in range(i + window_size, len(data) - window_size):
            window2 = data[j:j+window_size]
            
            if np.array_equal(window1, window2):
                elapsed = time.time() - start
                print(f"CICLO DETECTADO!")
                print(f"    Posição 1: {i}")
                print(f"    Posição 2: {j}")
                print(f"    Distância: {j - i} valores ({(j-i)*4} bytes)")
                print(f"    Tempo: {elapsed:.2f}s")
                return {
                    'cycle_found': True,
                    'position1': i,
                    'position2': j,
                    'distance': j - i,
                    'time': elapsed
                }
        
        # Progress indicator
        if i % 100000 == 0 and i > 0:
            print(f"\r  Verificado: {i}/{len(data)} ({i/len(data)*100:.1f}%)", end="", flush=True)
    
    elapsed = time.time() - start
    print(f"\r  ✓ Verificação completa ({elapsed:.1f}s)")
    
    return {
        'cycle_found': False,
        'values_tested': len(data),
        'bytes_tested': len(data) * 4,
        'time': elapsed
    }

cycle_result = test_cycle_detection(data_file, max_values=10_000_000, window_size=100)

print()
if cycle_result['cycle_found']:
    print(f"❌ CICLO DETECTADO em {cycle_result['distance']} valores")
    print(f"   Período estimado: 2^{np.log2(cycle_result['distance']):.1f}")
else:
    print(f"✅ NENHUM CICLO DETECTADO")
    print(f"   Valores testados: {cycle_result['values_tested']:,} ({cycle_result['bytes_tested']/1e6:.1f} MB)")
    print(f"   Período estimado: > 2^{np.log2(cycle_result['values_tested']):.1f}")

print()

# ============================================================================
# TESTE 3: LINEAR CORRELATION
# ============================================================================

print("TESTE 3: LINEAR CORRELATION")
print("-" * 70)
print()

def test_linear_correlation(file_path, max_values=1_000_000):
    """Testa correlação linear entre valores consecutivos"""
    
    print(f"  Carregando {max_values} valores uint64...", end=" ", flush=True)
    data = np.fromfile(file_path, dtype=np.uint64, count=max_values)
    print(f"{len(data)} carregados")
    
    print(f"  Calculando correlações...", end=" ", flush=True)
    
    # Normalizar para [0, 1]
    data_norm = data.astype(np.float64) / (2**64 - 1)
    
    results = {}
    
    # Lag 1 (valores consecutivos)
    corr_lag1 = np.corrcoef(data_norm[:-1], data_norm[1:])[0, 1]
    results['lag1'] = corr_lag1
    
    # Lag 2
    corr_lag2 = np.corrcoef(data_norm[:-2], data_norm[2:])[0, 1]
    results['lag2'] = corr_lag2
    
    # Lag 10
    corr_lag10 = np.corrcoef(data_norm[:-10], data_norm[10:])[0, 1]
    results['lag10'] = corr_lag10
    
    # Lag 100
    corr_lag100 = np.corrcoef(data_norm[:-100], data_norm[100:])[0, 1]
    results['lag100'] = corr_lag100
    
    print("✓")
    
    return results

corr_results = test_linear_correlation(data_file)

print()
print("Correlações lineares:")
for lag, corr in corr_results.items():
    abs_corr = abs(corr)
    status = "✅ EXCELENTE" if abs_corr < 0.01 else "⚠️ DETECTÁVEL" if abs_corr < 0.05 else "❌ ALTO"
    print(f"  {lag}: {corr:+.6f} (|r| = {abs_corr:.6f}) {status}")

print()

# ============================================================================
# TESTE 4: SPECTRAL ENTROPY
# ============================================================================

print("TESTE 4: SPECTRAL ENTROPY")
print("-" * 70)
print()

def test_spectral_entropy(file_path, max_values=100_000):
    """Calcula entropia espectral via FFT"""
    
    print(f"  Carregando {max_values} valores uint8...", end=" ", flush=True)
    data = np.fromfile(file_path, dtype=np.uint8, count=max_values)
    print(f"{len(data)} carregados")
    
    print(f"  Calculando FFT...", end=" ", flush=True)
    
    # Converter para [-1, 1]
    data_norm = (data.astype(np.float64) - 127.5) / 127.5
    
    # FFT
    fft = np.fft.fft(data_norm)
    power_spectrum = np.abs(fft)**2
    
    # Normalizar power spectrum
    power_spectrum = power_spectrum / np.sum(power_spectrum)
    
    # Calcular entropia espectral
    spectral_entropy = 0.0
    for p in power_spectrum:
        if p > 0:
            spectral_entropy -= p * np.log2(p)
    
    # Entropia máxima (distribuição uniforme)
    max_spectral_entropy = np.log2(len(power_spectrum))
    
    # Normalizar
    normalized_spectral = spectral_entropy / max_spectral_entropy
    
    print("✓")
    
    # Análise de picos no espectro
    threshold = np.mean(power_spectrum) + 3 * np.std(power_spectrum)
    peaks = np.sum(power_spectrum > threshold)
    
    return {
        'entropy': spectral_entropy,
        'max_entropy': max_spectral_entropy,
        'normalized': normalized_spectral,
        'peaks': peaks,
        'mean_power': np.mean(power_spectrum),
        'std_power': np.std(power_spectrum)
    }

spectral_result = test_spectral_entropy(data_file)

print()
print("Entropia Espectral:")
print(f"  Entropia: {spectral_result['entropy']:.2f} / {spectral_result['max_entropy']:.2f}")
print(f"  Normalizada: {spectral_result['normalized']*100:.2f}%")
print(f"  Picos detectados: {spectral_result['peaks']}")

status = "✅ EXCELENTE" if spectral_result['normalized'] > 0.95 else "⚠️ BOM" if spectral_result['normalized'] > 0.90 else "❌ FRACO"
print(f"  Status: {status}")

print()

# ============================================================================
# RESUMO FINAL
# ============================================================================

print("="*70)
print("RESUMO FINAL")
print("="*70)
print()

print("AGLE Pure (Avalanche Encadeado) - Resultados:")
print()

# N-gram
avg_predictability = np.mean([r['predictability'] for r in ngram_results.values()])
ngram_status = "✅ APROVADO" if avg_predictability < 0.01 else "⚠️ BORDERLINE" if avg_predictability < 0.05 else "❌ REPROVADO"
print(f"1. N-gram Predictability: {avg_predictability*100:.4f}% {ngram_status}")

# Cycle
cycle_status = "✅ APROVADO" if not cycle_result['cycle_found'] else "❌ REPROVADO"
print(f"2. Cycle Detection: {cycle_status}")
if not cycle_result['cycle_found']:
    print(f"   Período mínimo: > 2^{np.log2(cycle_result['values_tested']):.1f}")

# Correlation
max_corr = max(abs(c) for c in corr_results.values())
corr_status = "✅ APROVADO" if max_corr < 0.01 else "⚠️ BORDERLINE" if max_corr < 0.05 else "❌ REPROVADO"
print(f"3. Linear Correlation: max |r| = {max_corr:.6f} {corr_status}")

# Spectral
spectral_status = "✅ APROVADO" if spectral_result['normalized'] > 0.95 else "⚠️ BORDERLINE" if spectral_result['normalized'] > 0.90 else "❌ REPROVADO"
print(f"4. Spectral Entropy: {spectral_result['normalized']*100:.2f}% {spectral_status}")

print()

# Conclusão geral
all_pass = (
    avg_predictability < 0.01 and
    not cycle_result['cycle_found'] and
    max_corr < 0.01 and
    spectral_result['normalized'] > 0.95
)

if all_pass:
    print("✅ CONCLUSÃO: AGLE Pure APROVADO em todos os testes")
    print("   Avalanche encadeado demonstra excelente difusão")
else:
    print("⚠️ CONCLUSÃO: Alguns testes necessitam revisão")

print()
