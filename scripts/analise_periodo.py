import numpy as np
import sys

# Lê arquivo binário de uint32_t e mede período de repetição
filename = sys.argv[1] if len(sys.argv) > 1 else 'rctm_uint32.bin'
N = 5000000
with open(filename, 'rb') as f:
    data = np.frombuffer(f.read(N*4), dtype=np.uint32)

# 1️⃣ Medir período real (primeira repetição de bloco)
seen = {}
period = -1
for i, val in enumerate(data):
    if val in seen:
        period = i - seen[val]
        break
    seen[val] = i
with open('/home/andre/Documentos/resultado/periodo_real.txt', 'w') as f:
    if period > 0:
        f.write(f'Primeira repetição após {period} passos\n')
    else:
        f.write('Nenhuma repetição detectada no intervalo\n')

# 2️⃣ Distribuição de estados repetidos
from collections import Counter
counts = Counter(data)
with open('/home/andre/Documentos/resultado/estados_repetidos.txt', 'w') as f:
    for val, cnt in counts.most_common(20):
        if cnt > 1:
            f.write(f'Valor {val} repetido {cnt} vezes\n')

# 3️⃣ Entropia condicional (aproximação)
def conditional_entropy(seq, k=1):
    from collections import Counter
    pairs = zip(seq[:-k], seq[k:])
    pair_counts = Counter(pairs)
    total = sum(pair_counts.values())
    import math
    H = 0.0
    for (a, b), cnt in pair_counts.items():
        pa = seq.count(a) / len(seq)
        pab = cnt / total
        if pab > 0 and pa > 0:
            H -= pab * math.log2(pab / pa)
    return H
H1 = conditional_entropy(list(data), k=1)
with open('/home/andre/Documentos/resultado/entropia_condicional.txt', 'w') as f:
    f.write(f'Entropia condicional (k=1): {H1:.6f} bits\n')
