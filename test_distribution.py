#!/usr/bin/env python3
"""
Script para testar a distribuição dos números gerados por agle_v3_experimental
"""

import subprocess
import struct
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def generate_random_data(n_samples=10000, seed=123456789):
    """Gera dados aleatórios do programa C"""
    cmd = [
        './versoes/v2/codigo/agle_v3_experimental',
        '--stdout',
        str(seed)
    ]
    
    print(f"Gerando {n_samples} amostras...")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd='/home/andre/Imagens/mestrado/Hardware-Induced-Irreversibility-in-Chaotic-Maps-main'
    )
    
    data = []
    bytes_read = 0
    
    try:
        while len(data) < n_samples:
            output = process.stdout.read(8)  # uint64_t = 8 bytes
            if not output:
                break
            value = struct.unpack('<Q', output)[0]  # unsigned 64-bit
            data.append(value)
            bytes_read += 8
    finally:
        process.terminate()
        process.wait()
    
    return np.array(data, dtype=np.uint64)

def normalize_to_float(data):
    """Normaliza uint64 para [0, 1)"""
    return data / (2**64)

def test_distribution(data):
    """Testa propriedades estatísticas da distribuição"""
    
    print("\n" + "="*60)
    print("ANÁLISE DE DISTRIBUIÇÃO")
    print("="*60)
    
    # Estatísticas básicas
    print(f"\nEstatísticas básicas:")
    print(f"  Amostras: {len(data)}")
    print(f"  Mín: {data.min()}")
    print(f"  Máx: {data.max()}")
    print(f"  Média: {data.mean():.6e}")
    print(f"  Desvio padrão: {data.std():.6e}")
    print(f"  Mediana: {np.median(data):.6e}")
    
    # Normalizar para [0, 1)
    norm_data = normalize_to_float(data)
    
    print(f"\nEstatísticas normalizadas [0,1):")
    print(f"  Média: {norm_data.mean():.10f} (esperado: 0.5)")
    print(f"  Desvio padrão: {norm_data.std():.10f} (esperado: 0.2887)")
    
    # Teste de uniformidade (Kolmogorov-Smirnov)
    ks_stat, ks_pval = stats.kstest(norm_data, 'uniform')
    print(f"\nTeste Kolmogorov-Smirnov (uniformidade):")
    print(f"  Estatística: {ks_stat:.6f}")
    print(f"  P-valor: {ks_pval:.6f}")
    print(f"  Resultado: {'PASSOU' if ks_pval > 0.05 else 'FALHOU'} (esperado p > 0.05)")
    
    # Teste Chi-square
    hist, bin_edges = np.histogram(norm_data, bins=100)
    expected_count = len(data) / 100
    chi2 = np.sum((hist - expected_count)**2 / expected_count)
    chi2_pval = 1 - stats.chi2.cdf(chi2, df=99)
    print(f"\nTeste Chi-square (uniformidade):")
    print(f"  Chi2: {chi2:.6f}")
    print(f"  P-valor: {chi2_pval:.6f}")
    print(f"  Resultado: {'PASSOU' if chi2_pval > 0.05 else 'FALHOU'} (esperado p > 0.05)")
    
    # Teste de autocorrelação
    if len(data) > 1000:
        autocorr = np.corrcoef(norm_data[:-1], norm_data[1:])[0,1]
        print(f"\nAutocorrelação (lag-1):")
        print(f"  Correlação: {autocorr:.6f}")
        print(f"  Resultado: {'BOA' if abs(autocorr) < 0.05 else 'RUIM'} (esperado ~0)")
    
    # Entropia
    entropy = -np.sum((hist/len(data)) * np.log2((hist+1e-10)/len(data)))
    max_entropy = -np.sum((np.ones(100)/100) * np.log2(1/100))
    print(f"\nEntropia:")
    print(f"  Entropia observada: {entropy:.6f}")
    print(f"  Entropia máxima: {max_entropy:.6f}")
    print(f"  Taxa de utilização: {entropy/max_entropy*100:.2f}%")
    
    return norm_data, hist, bin_edges

def plot_distribution(norm_data, hist, bin_edges):
    """Visualiza a distribuição"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Histogram
    ax = axes[0, 0]
    ax.bar((bin_edges[:-1] + bin_edges[1:]) / 2, hist, width=bin_edges[1]-bin_edges[0])
    ax.set_xlabel('Valor normalizado [0, 1)')
    ax.set_ylabel('Frequência')
    ax.set_title('Histograma de frequências (100 bins)')
    ax.grid(True, alpha=0.3)
    
    # CDF vs uniforme teórica
    ax = axes[0, 1]
    sorted_data = np.sort(norm_data)
    empirical_cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
    ax.plot(sorted_data, empirical_cdf, label='CDF observada', linewidth=2)
    ax.plot([0, 1], [0, 1], 'r--', label='CDF teórica (uniforme)', linewidth=2)
    ax.set_xlabel('Valor')
    ax.set_ylabel('Probabilidade cumulativa')
    ax.set_title('Função de distribuição cumulativa')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Q-Q plot
    ax = axes[1, 0]
    stats.probplot(norm_data, dist="uniform", plot=ax)
    ax.set_title('Q-Q Plot (uniforme)')
    ax.grid(True, alpha=0.3)
    
    # Sequência temporal (primeiros 1000 valores)
    ax = axes[1, 1]
    ax.plot(norm_data[:1000], linewidth=0.5)
    ax.set_xlabel('Índice')
    ax.set_ylabel('Valor normalizado')
    ax.set_title('Sequência temporal (primeiros 1000 valores)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/andre/Imagens/mestrado/Hardware-Induced-Irreversibility-in-Chaotic-Maps-main/resultados/analises/distribution_test_v3.png', dpi=100)
    print(f"\nGráfico salvo em: resultados/analises/distribution_test_v3.png")
    plt.close()

if __name__ == '__main__':
    # Gerar dados
    data = generate_random_data(n_samples=10000, seed=123456789)
    
    # Testar distribuição
    norm_data, hist, bin_edges = test_distribution(data)
    
    # Visualizar
    plot_distribution(norm_data, hist, bin_edges)
    
    print("\n" + "="*60)
    print("Teste de distribuição concluído!")
    print("="*60)
