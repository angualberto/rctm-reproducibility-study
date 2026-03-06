#!/bin/bash

# Limpa arquivos antigos
rm -f rctm_acumulado.bin log_geracao.txt

echo "Iniciando geração de dados acumulados (10 iterações)..."

for i in {1..10}; do
    echo "=== Iteração $i/10 ===" | tee -a log_geracao.txt
    
    # Remove arquivo anterior se existir
    rm -f rctm61.81.bin
    
    # Gera novos dados
    octave --no-gui --quiet main.m
    
    # Verifica se foi gerado com sucesso
    if [ -f rctm61.81.bin ]; then
        SIZE=$(stat -c%s rctm61.81.bin)
        echo "Arquivo gerado: $SIZE bytes" | tee -a log_geracao.txt
        
        # Acumula os dados
        cat rctm61.81.bin >> rctm_acumulado.bin
    else
        echo "ERRO: rctm61.81.bin não foi gerado!" | tee -a log_geracao.txt
        exit 1
    fi
done

# Verifica o tamanho final
FINAL_SIZE=$(stat -c%s rctm_acumulado.bin 2>/dev/null || echo 0)
echo "=== Geração completa! ===" | tee -a log_geracao.txt
echo "Tamanho total: $FINAL_SIZE bytes (~$(($FINAL_SIZE / 1024 / 1024))MB)" | tee -a log_geracao.txt

# Agora roda o dieharder
echo "Iniciando testes dieharder..." | tee -a log_geracao.txt
dieharder -g 201 -f rctm_acumulado.bin -a > resultado_oficial_validado.txt

echo "Teste concluído! Resultados em resultado_oficial_validado.txt" | tee -a log_geracao.txt
