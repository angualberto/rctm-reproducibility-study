#!/usr/bin/env python3
"""
Teste de Predibilidade - AGLE Parallel

Tenta prever próximos valores usando:
1. Correlação linear (LAG-N)
2. Análise de autocorrelação
3. Regressão linear simples
4. Rede neural LSTM (se disponível)
"""

import sys
import struct
import subprocess
import numpy as np
from collections import defaultdict

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("⚠️  scipy não disponível - alguns testes serão pulados")

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠️  PyTorch não disponível - teste LSTM será pulado")


def read_rng_samples(agle_bin, seed, num_samples):
    """Lê samples do gerador AGLE"""
    cmd = [agle_bin, '--stdout', str(seed)]
    
    print(f"🔄 Gerando {num_samples:,} samples...")
    
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    
    samples = []
    bytes_needed = num_samples * 8
    data = proc.stdout.read(bytes_needed)
    
    proc.terminate()
    proc.wait()
    
    for i in range(0, len(data), 8):
        if i + 8 <= len(data):
            value = struct.unpack('<Q', data[i:i+8])[0]
            samples.append(value)
    
    return np.array(samples, dtype=np.uint64)


def test_linear_correlation(samples, max_lag=20):
    """Testa correlação linear entre valores consecutivos"""
    print("\n" + "="*50)
    print("📊 Teste 1: Correlação Linear (LAG-N)")
    print("="*50)
    
    # Converter para float normalizado
    samples_norm = samples.astype(np.float64) / (2**64)
    
    results = []
    
    for lag in range(1, max_lag + 1):
        x = samples_norm[:-lag]
        y = samples_norm[lag:]
        
        if HAS_SCIPY:
            corr, p_value = stats.pearsonr(x, y)
        else:
            # Correlação manual
            mean_x = np.mean(x)
            mean_y = np.mean(y)
            cov = np.mean((x - mean_x) * (y - mean_y))
            std_x = np.std(x)
            std_y = np.std(y)
            corr = cov / (std_x * std_y + 1e-10)
            p_value = -1
        
        results.append((lag, corr, p_value))
        
        if lag <= 5 or abs(corr) > 0.01:
            if HAS_SCIPY:
                print(f"  LAG-{lag:2d}: corr={corr:+.6f}, p-value={p_value:.3e}")
            else:
                print(f"  LAG-{lag:2d}: corr={corr:+.6f}")
    
    max_corr = max(abs(r[1]) for r in results)
    
    print(f"\n  Correlação máxima: {max_corr:.6f}")
    
    if max_corr < 0.01:
        print("  ✅ PASSOU - Correlação desprezível")
        return True
    elif max_corr < 0.05:
        print("  ⚠️  SUSPEITO - Correlação fraca detectada")
        return False
    else:
        print("  ❌ FALHOU - Correlação significativa!")
        return False


def test_autocorrelation(samples, max_lag=50):
    """Testa autocorrelação nos bits"""
    print("\n" + "="*50)
    print("📊 Teste 2: Autocorrelação de Bits")
    print("="*50)
    
    # Extrair bits individuais (LSB)
    bits = samples & 1
    bits_float = bits.astype(np.float64)
    bits_centered = bits_float - np.mean(bits_float)
    
    autocorr = []
    
    for lag in range(1, max_lag + 1):
        x = bits_centered[:-lag]
        y = bits_centered[lag:]
        
        ac = np.sum(x * y) / (len(x) * np.var(bits_float))
        autocorr.append(ac)
        
        if lag <= 10 or abs(ac) > 0.01:
            print(f"  LAG-{lag:2d}: autocorr={ac:+.6f}")
    
    max_autocorr = max(abs(a) for a in autocorr)
    
    print(f"\n  Autocorrelação máxima: {max_autocorr:.6f}")
    
    if max_autocorr < 0.01:
        print("  ✅ PASSOU - Sem autocorrelação detectável")
        return True
    else:
        print("  ❌ FALHOU - Autocorrelação significativa!")
        return False


def test_linear_prediction(samples, window=10, test_size=1000):
    """Tenta prever próximo valor usando regressão linear"""
    print("\n" + "="*50)
    print("📊 Teste 3: Predição Linear (Janela={})".format(window))
    print("="*50)
    
    # Normalizar
    samples_norm = samples.astype(np.float64) / (2**64)
    
    # Preparar dados de treino e teste
    train_size = len(samples_norm) - test_size - window
    
    X_train = []
    y_train = []
    
    for i in range(train_size):
        X_train.append(samples_norm[i:i+window])
        y_train.append(samples_norm[i+window])
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Treinar regressão linear simples
    # y = w * X + b
    X_mean = np.mean(X_train, axis=0)
    y_mean = np.mean(y_train)
    
    # Calcular coeficientes
    numerator = np.sum((X_train - X_mean) * (y_train - y_mean).reshape(-1, 1), axis=0)
    denominator = np.sum((X_train - X_mean) ** 2, axis=0)
    weights = numerator / (denominator + 1e-10)
    bias = y_mean - np.sum(weights * X_mean)
    
    # Testar predição
    X_test = []
    y_test = []
    
    for i in range(train_size, len(samples_norm) - window):
        X_test.append(samples_norm[i:i+window])
        y_test.append(samples_norm[i+window])
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    # Predizer
    y_pred = np.sum(X_test * weights, axis=1) + bias
    
    # Calcular erro
    mse = np.mean((y_test - y_pred) ** 2)
    mae = np.mean(np.abs(y_test - y_pred))
    
    # Baseline: predizer média
    baseline_mse = np.mean((y_test - y_mean) ** 2)
    
    improvement = (baseline_mse - mse) / baseline_mse * 100
    
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  Baseline MSE: {baseline_mse:.6f}")
    print(f"  Melhoria: {improvement:+.2f}%")
    
    if improvement < 1.0:
        print("  ✅ PASSOU - Sem predição linear detectável")
        return True
    elif improvement < 5.0:
        print("  ⚠️  SUSPEITO - Pequena melhoria sobre baseline")
        return False
    else:
        print("  ❌ FALHOU - Predição linear funciona!")
        return False


def test_lstm_prediction(samples, window=32, test_size=5000):
    """Tenta prever próximo valor usando LSTM"""
    if not HAS_TORCH:
        print("\n⚠️  Teste LSTM pulado (PyTorch não disponível)")
        return True
    
    print("\n" + "="*50)
    print("📊 Teste 4: Predição LSTM (Janela={})".format(window))
    print("="*50)
    
    # Normalizar
    samples_norm = samples.astype(np.float32) / (2**64)
    
    # Preparar dados
    train_size = len(samples_norm) - test_size - window
    
    X_train = []
    y_train = []
    
    for i in range(train_size):
        X_train.append(samples_norm[i:i+window])
        y_train.append(samples_norm[i+window])
    
    X_train = torch.FloatTensor(X_train).unsqueeze(-1)  # (N, window, 1)
    y_train = torch.FloatTensor(y_train)
    
    X_test = []
    y_test = []
    
    for i in range(train_size, len(samples_norm) - window):
        X_test.append(samples_norm[i:i+window])
        y_test.append(samples_norm[i+window])
    
    X_test = torch.FloatTensor(X_test).unsqueeze(-1)
    y_test = torch.FloatTensor(y_test)
    
    # Modelo LSTM simples
    class SimpleLSTM(nn.Module):
        def __init__(self, input_size=1, hidden_size=32):
            super(SimpleLSTM, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)
        
        def forward(self, x):
            _, (h_n, _) = self.lstm(x)
            out = self.fc(h_n[-1])
            return out.squeeze()
    
    model = SimpleLSTM()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("  Treinando LSTM...")
    
    # Treinar
    num_epochs = 10
    batch_size = 128
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 2 == 0:
            avg_loss = total_loss / (len(X_train) // batch_size)
            print(f"    Época {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
    
    # Testar
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).numpy()
        y_test_np = y_test.numpy()
        
        mse = np.mean((y_test_np - y_pred) ** 2)
        mae = np.mean(np.abs(y_test_np - y_pred))
        
        # Baseline: predizer média
        baseline_mse = np.var(y_test_np)
        
        improvement = (baseline_mse - mse) / baseline_mse * 100
    
    print(f"\n  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  Baseline MSE: {baseline_mse:.6f}")
    print(f"  Melhoria: {improvement:+.2f}%")
    
    if improvement < 2.0:
        print("  ✅ PASSOU - LSTM não consegue prever")
        return True
    elif improvement < 10.0:
        print("  ⚠️  SUSPEITO - LSTM tem pequeno sucesso")
        return False
    else:
        print("  ❌ FALHOU - LSTM consegue prever!")
        return False


def test_bit_frequency(samples):
    """Testa frequência de bits individuais"""
    print("\n" + "="*50)
    print("📊 Teste 5: Frequência de Bits")
    print("="*50)
    
    bit_counts = np.zeros(64, dtype=np.uint64)
    
    for sample in samples:
        sample = int(sample)  # Converter para Python int
        for bit in range(64):
            if (sample >> bit) & 1:
                bit_counts[bit] += 1
    
    expected = len(samples) / 2.0
    max_deviation = 0
    
    for bit in range(64):
        deviation = abs(bit_counts[bit] - expected) / expected * 100
        max_deviation = max(max_deviation, deviation)
        
        if deviation > 1.0:
            print(f"  Bit {bit:2d}: {bit_counts[bit]:8d} / {len(samples)} ({deviation:+.2f}%)")
    
    print(f"\n  Desvio máximo: {max_deviation:.2f}%")
    
    if max_deviation < 1.0:
        print("  ✅ PASSOU - Frequências balanceadas")
        return True
    elif max_deviation < 2.0:
        print("  ⚠️  SUSPEITO - Pequeno desbalanceamento")
        return False
    else:
        print("  ❌ FALHOU - Desbalanceamento significativo!")
        return False


def main():
    if len(sys.argv) < 2:
        print("Uso: python3 test_prediction.py <agle_binary> [seed] [num_samples]")
        print("\nExemplo:")
        print("  python3 test_prediction.py ./agle_parallel 123456789 50000")
        sys.exit(1)
    
    agle_bin = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) >= 3 else 123456789
    num_samples = int(sys.argv[3]) if len(sys.argv) >= 4 else 50000
    
    print("="*50)
    print("🔍 TESTE DE PREDIBILIDADE - AGLE PARALLEL")
    print("="*50)
    print(f"Gerador: {agle_bin}")
    print(f"Seed: {seed}")
    print(f"Samples: {num_samples:,}")
    print("="*50)
    
    # Ler samples
    samples = read_rng_samples(agle_bin, seed, num_samples)
    
    if len(samples) < num_samples:
        print(f"\n❌ Erro: apenas {len(samples)} samples lidos")
        sys.exit(1)
    
    print(f"✅ {len(samples):,} samples lidos com sucesso\n")
    
    # Executar testes
    results = []
    
    results.append(("Correlação Linear", test_linear_correlation(samples)))
    results.append(("Autocorrelação", test_autocorrelation(samples)))
    results.append(("Predição Linear", test_linear_prediction(samples)))
    results.append(("Predição LSTM", test_lstm_prediction(samples)))
    results.append(("Frequência de Bits", test_bit_frequency(samples)))
    
    # Resumo
    print("\n" + "="*50)
    print("📋 RESUMO DOS TESTES")
    print("="*50)
    
    passed = 0
    failed = 0
    
    for name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"  {name:25s}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print("="*50)
    print(f"Total: {passed}/{len(results)} testes passaram")
    
    if failed == 0:
        print("\n🎉 EXCELENTE - Nenhuma predição detectável!")
        return 0
    elif failed <= 2:
        print("\n⚠️  ATENÇÃO - Alguns padrões detectados")
        return 1
    else:
        print("\n❌ FALHA - Gerador é predizível!")
        return 2


if __name__ == '__main__':
    sys.exit(main())
