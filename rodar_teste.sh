#!/bin/bash

# Configurações
ARQUIVO_BIN="rctm_grande.bin"
LOG_FINAL="resultado_$(hostname)_$(date +%Y%m%d).txt"

echo "=========================================================="
echo "Iniciando Geração de Dados para Mestrado - Andre Gualberto"
echo "=========================================================="

# 1. Gerar o arquivo binário grande via Octave
# Nota: Certifique-se que seu main.m gera saída bruta para o stdout ou para o arquivo
echo "[1/2] Gerando 500MB de dados (isso pode demorar)..."
# Exemplo se o main.m aceitar argumentos ou você pode editar o main.m para aumentar o loop
octave --no-gui --quiet main.m > $ARQUIVO_BIN

TAMANHO=$(ls -lh $ARQUIVO_BIN | awk '{print $5}')
echo "Arquivo gerado com sucesso: $TAMANHO"

# 2. Rodar o Dieharder com o Gerador 201 (Raw Binary)
echo "[2/2] Iniciando Bateria Completa Dieharder..."
dieharder -g 201 -f $ARQUIVO_BIN -a > $LOG_FINAL

echo "=========================================================="
echo "Teste Concluído! Resultado salvo em: $LOG_FINAL"
echo "=========================================================="
