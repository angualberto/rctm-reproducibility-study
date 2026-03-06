#!/usr/bin/env python3
"""
Consolida todos os resultados de testes AGLE V3 Harmonic
Gera gráficos e tabela consolidada
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re

# Configurações
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10
plt.style.use('seaborn-v0_8-darkgrid')

# Diretório de resultados
RESULT_DIR = Path("resultados/analises")

# Estrutura de dados para armazenar resultados
results = {
    'Transformer Distinguisher': {'status': '✅ PASSOU', 'score': 32.13, 'metric': 'Accuracy (lower=better)', 'baseline': 33.33},
    'State Recovery': {'status': '✅ PASSOU', 'score': 0, 'metric': 'Repetitions', 'baseline': 0},
    'Shannon Entropy': {'status': '✅ PASSOU', 'score': 7.999979, 'metric': 'bits/byte', 'baseline': 8.0},
    'Cycle Detection': {'status': '❌ FALHOU', 'score': 111332, 'metric': 'Unique values', 'baseline': 1000000},
    'Permutation Entropy (ord 5)': {'status': '✅ EXCELENTE', 'score': 0.999955, 'metric': 'Normalized', 'baseline': 1.0},
    'ML Predictability': {'status': '✅ PASSOU', 'score': 0.388, 'metric': 'Accuracy %', 'baseline': 0.391},
    'N-gram 2-gram': {'status': '⚠️ AVISO', 'score': 1.113, 'metric': 'Accuracy %', 'baseline': 0.391},
    'N-gram 3-gram': {'status': '❌ FALHOU', 'score': 31.795, 'metric': 'Accuracy %', 'baseline': 0.391},
    'N-gram 4-gram': {'status': '❌ FALHOU', 'score': 99.405, 'metric': 'Accuracy %', 'baseline': 0.391},
    'Symbolic Dynamics': {'status': '✅ PASSOU', 'score': 0.999883, 'metric': 'Normalized entropy', 'baseline': 1.0},
    'Distinguishing Attack': {'status': '❌ FALHOU', 'score': 0.550, 'metric': 'Normalized entropy', 'baseline': 1.0},
    'State Reconstruction': {'status': '✅ PASSOU', 'score': 0.03477, 'metric': 'Mean neighbor distance', 'baseline': 0.0},
    'Neural v2 Transformer': {'status': '✅ PASSOU', 'score': 32.72, 'metric': 'Accuracy %', 'baseline': 33.33},
    'Neural v2 LSTM': {'status': '✅ PASSOU', 'score': 34.11, 'metric': 'Accuracy %', 'baseline': 33.33},
    'Neural v2 ACF': {'status': '✅ PASSOU', 'score': 0.011125, 'metric': 'Autocorrelation', 'baseline': 0.0},
}

# Categorias
categories = {
    'Distinguishing Attacks': ['Transformer Distinguisher', 'Neural v2 Transformer', 'Neural v2 LSTM', 'Distinguishing Attack'],
    'Entropy Tests': ['Shannon Entropy', 'Permutation Entropy (ord 5)', 'Symbolic Dynamics'],
    'Predictability': ['ML Predictability', 'N-gram 2-gram', 'N-gram 3-gram', 'N-gram 4-gram'],
    'State Analysis': ['State Recovery', 'State Reconstruction', 'Cycle Detection', 'Neural v2 ACF'],
}

def create_summary_table():
    """Cria tabela consolidada de resultados"""
    
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepara dados da tabela
    table_data = []
    table_data.append(['Teste', 'Status', 'Score', 'Métrica', 'Baseline'])
    
    for test, data in results.items():
        table_data.append([
            test,
            data['status'],
            f"{data['score']:.4f}" if isinstance(data['score'], float) else str(data['score']),
            data['metric'],
            f"{data['baseline']:.4f}" if isinstance(data['baseline'], float) else str(data['baseline'])
        ])
    
    # Criar tabela
    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.35, 0.15, 0.15, 0.25, 0.10])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # Estilizar cabeçalho
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Colorir linhas alternadas
    for i in range(1, len(table_data)):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            
            # Colorir status
            if j == 1:
                status = table_data[i][1]
                if '✅' in status:
                    table[(i, j)].set_facecolor('#C8E6C9')
                elif '❌' in status:
                    table[(i, j)].set_facecolor('#FFCDD2')
                elif '⚠️' in status:
                    table[(i, j)].set_facecolor('#FFF9C4')
    
    plt.title('AGLE V3 Harmonic - Consolidação de Resultados de Testes\n2026-03-05', 
              fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig('resultados/analises/consolidated_table.png', dpi=300, bbox_inches='tight')
    print("✓ Tabela salva: resultados/analises/consolidated_table.png")
    plt.close()

def create_category_charts():
    """Cria gráficos por categoria"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('AGLE V3 Harmonic - Análise por Categoria\n2026-03-05', 
                 fontsize=16, fontweight='bold')
    
    for idx, (category, tests) in enumerate(categories.items()):
        ax = axes[idx // 2, idx % 2]
        
        # Filtrar testes da categoria
        cat_tests = [t for t in tests if t in results]
        scores = [results[t]['score'] for t in cat_tests]
        statuses = [results[t]['status'] for t in cat_tests]
        
        # Cor baseada no status
        colors = []
        for status in statuses:
            if '✅' in status:
                colors.append('#4CAF50')
            elif '❌' in status:
                colors.append('#F44336')
            elif '⚠️' in status:
                colors.append('#FFC107')
            else:
                colors.append('#9E9E9E')
        
        # Gráfico de barras
        y_pos = np.arange(len(cat_tests))
        bars = ax.barh(y_pos, scores, color=colors, alpha=0.7)
        
        # Labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels([t[:30] for t in cat_tests], fontsize=8)
        ax.set_xlabel('Score', fontsize=10)
        ax.set_title(category, fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Adicionar valores nas barras
        for i, (bar, score) in enumerate(zip(bars, scores)):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f' {score:.2f}',
                   ha='left', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('resultados/analises/category_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Gráficos por categoria salvos: resultados/analises/category_analysis.png")
    plt.close()

def create_pass_fail_chart():
    """Cria gráfico de aprovação/falha"""
    
    # Contar status
    passed = sum(1 for r in results.values() if '✅' in r['status'])
    failed = sum(1 for r in results.values() if '❌' in r['status'])
    warning = sum(1 for r in results.values() if '⚠️' in r['status'])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pizza
    labels = [f'✅ Passou\n({passed})', f'❌ Falhou\n({failed})', f'⚠️ Aviso\n({warning})']
    sizes = [passed, failed, warning]
    colors = ['#4CAF50', '#F44336', '#FFC107']
    explode = (0.1, 0, 0)
    
    axes[0].pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 12})
    axes[0].set_title('Distribuição de Resultados', fontsize=14, fontweight='bold')
    
    # Barras
    status_counts = {'Passed': passed, 'Failed': failed, 'Warning': warning}
    bars = axes[1].bar(status_counts.keys(), status_counts.values(), 
                       color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[1].set_ylabel('Número de Testes', fontsize=12)
    axes[1].set_title('Resumo de Status', fontsize=14, fontweight='bold')
    axes[1].set_ylim(0, max(status_counts.values()) + 2)
    axes[1].grid(axis='y', alpha=0.3)
    
    # Adicionar valores nas barras
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.suptitle('AGLE V3 Harmonic - Resumo de Aprovação\n2026-03-05',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('resultados/analises/pass_fail_summary.png', dpi=300, bbox_inches='tight')
    print("✓ Resumo de aprovação salvo: resultados/analises/pass_fail_summary.png")
    plt.close()

def create_score_comparison():
    """Cria gráfico comparativo de scores normalizados"""
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Normalizar scores para escala 0-1 (0 = falhou, 1 = passou perfeitamente)
    normalized_scores = {}
    
    for test, data in results.items():
        score = data['score']
        baseline = data['baseline']
        
        # Lógica de normalização depende do teste
        if 'Accuracy' in data['metric'] and 'lower=better' in data['metric']:
            # Distinguishers: menor é melhor (próximo de baseline)
            if baseline > 0:
                normalized = 1.0 - abs(score - baseline) / baseline
            else:
                normalized = 0.5
        elif 'Accuracy' in data['metric'] and 'gram' in test:
            # N-gram: menor é melhor
            if score <= 0.6:
                normalized = 1.0
            elif score <= 1.2:
                normalized = 0.7
            elif score <= 2.0:
                normalized = 0.4
            else:
                normalized = max(0.0, 1.0 - (score / 100.0))
        elif 'Normalized' in data['metric']:
            # Entropy tests: próximo de 1.0 é melhor
            normalized = score
        elif 'Repetitions' in data['metric']:
            # State recovery: 0 é melhor
            normalized = 1.0 if score == 0 else 0.0
        elif 'Unique values' in data['metric']:
            # Cycle detection: maior é melhor
            normalized = min(1.0, score / baseline)
        elif 'distance' in data['metric']:
            # State reconstruction: próximo de 0 é melhor
            normalized = max(0.0, 1.0 - score)
        elif 'Autocorrelation' in data['metric']:
            # ACF: próximo de 0 é melhor
            normalized = max(0.0, 1.0 - abs(score))
        else:
            normalized = 0.5  # Padrão
        
        normalized_scores[test] = max(0.0, min(1.0, normalized))
    
    # Ordenar por score
    sorted_tests = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)
    tests, scores = zip(*sorted_tests)
    
    # Cores baseadas no score normalizado
    colors = []
    for score in scores:
        if score >= 0.9:
            colors.append('#4CAF50')  # Verde
        elif score >= 0.7:
            colors.append('#8BC34A')  # Verde claro
        elif score >= 0.5:
            colors.append('#FFC107')  # Amarelo
        elif score >= 0.3:
            colors.append('#FF9800')  # Laranja
        else:
            colors.append('#F44336')  # Vermelho
    
    # Gráfico
    y_pos = np.arange(len(tests))
    bars = ax.barh(y_pos, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([t[:40] for t in tests], fontsize=9)
    ax.set_xlabel('Score Normalizado (0=falhou, 1=perfeito)', fontsize=12, fontweight='bold')
    ax.set_title('AGLE V3 Harmonic - Comparação de Performance por Teste\n2026-03-05',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim(0, 1.1)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Linha de referência em 0.7 (mínimo aceitável)
    ax.axvline(x=0.7, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Mínimo Aceitável')
    ax.legend(fontsize=10)
    
    # Adicionar valores nas barras
    for i, (bar, score) in enumerate(zip(bars, scores)):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
               f'{score:.3f}',
               ha='left', va='center', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('resultados/analises/score_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Comparação de scores salva: resultados/analises/score_comparison.png")
    plt.close()

def generate_text_report():
    """Gera relatório em texto"""
    
    report = []
    report.append("="*80)
    report.append("AGLE V3 HARMONIC - RELATÓRIO CONSOLIDADO DE TESTES")
    report.append("Data: 2026-03-05")
    report.append("="*80)
    report.append("")
    
    # Resumo
    passed = sum(1 for r in results.values() if '✅' in r['status'])
    failed = sum(1 for r in results.values() if '❌' in r['status'])
    warning = sum(1 for r in results.values() if '⚠️' in r['status'])
    total = len(results)
    
    report.append(f"RESUMO GERAL:")
    report.append(f"  Total de testes: {total}")
    report.append(f"  ✅ Passou: {passed} ({100*passed/total:.1f}%)")
    report.append(f"  ❌ Falhou: {failed} ({100*failed/total:.1f}%)")
    report.append(f"  ⚠️ Aviso: {warning} ({100*warning/total:.1f}%)")
    report.append("")
    
    # Por categoria
    for category, tests in categories.items():
        report.append(f"\n{category}:")
        report.append("-" * 60)
        for test in tests:
            if test in results:
                data = results[test]
                report.append(f"  {test:35s} {data['status']:15s} {data['score']:.6f}")
    
    report.append("\n" + "="*80)
    report.append("CONCLUSÕES:")
    report.append("="*80)
    report.append("")
    report.append("✅ PONTOS FORTES:")
    report.append("  - Transformer/LSTM indistinguível de ChaCha20 e /dev/urandom")
    report.append("  - Shannon entropy próximo do máximo teórico (7.999979/8.0)")
    report.append("  - Permutation entropy excelente (0.999955)")
    report.append("  - ML predictability abaixo do baseline (0.388% vs 0.391%)")
    report.append("  - State recovery: sem repetições detectadas")
    report.append("  - State reconstruction: sem estrutura recuperável")
    report.append("  - Symbolic dynamics: sem padrões detectados")
    report.append("")
    report.append("❌ PONTOS FRACOS:")
    report.append("  - N-gram 3 e 4: alta previsibilidade (31.8% e 99.4%)")
    report.append("  - Cycle detection: repetição prematura em posição 111,332")
    report.append("  - Distinguishing attack: entropia normalizada baixa (0.550)")
    report.append("")
    report.append("⚠️ RECOMENDAÇÕES:")
    report.append("  - Investigar alta previsibilidade em n-gramas de ordem 3+")
    report.append("  - Melhorar mistura para evitar ciclos prematuros em uint32")
    report.append("  - Considerar aumentar tamanho do estado interno")
    report.append("  - Re-testar com saída uint64 em vez de uint32")
    report.append("")
    
    # Salvar relatório
    report_path = RESULT_DIR / "consolidated_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"✓ Relatório de texto salvo: {report_path}")
    
    # Imprimir na tela também
    print("\n" + '\n'.join(report))

def main():
    """Função principal"""
    print("="*80)
    print("CONSOLIDANDO RESULTADOS DE TESTES AGLE V3 HARMONIC")
    print("="*80)
    print()
    
    # Criar diretório se não existir
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Gerar visualizações
    print("Gerando tabela consolidada...")
    create_summary_table()
    
    print("\nGerando gráficos por categoria...")
    create_category_charts()
    
    print("\nGerando resumo de aprovação...")
    create_pass_fail_chart()
    
    print("\nGerando comparação de scores...")
    create_score_comparison()
    
    print("\nGerando relatório de texto...")
    generate_text_report()
    
    print("\n" + "="*80)
    print("✅ CONSOLIDAÇÃO COMPLETA!")
    print("="*80)
    print("\nArquivos gerados:")
    print("  - resultados/analises/consolidated_table.png")
    print("  - resultados/analises/category_analysis.png")
    print("  - resultados/analises/pass_fail_summary.png")
    print("  - resultados/analises/score_comparison.png")
    print("  - resultados/analises/consolidated_report.txt")
    print()

if __name__ == "__main__":
    main()
