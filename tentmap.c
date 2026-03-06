#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

// Função tent map
void tentmap(double x0, double m, uint32_t *out, size_t N) {
    double x = x0;
    for (size_t i = 0; i < N; ++i) {
        out[i] = (uint32_t)(x * 4294967296.0); // 2^32
        if (m <= 2.0) {
            if (x < 0.5)
                x = m * x;
            else
                x = m * (1.0 - x);
        } else {
            double n1 = 0.5 - fmod(m/2.0,1.0)/m;
            double n2 = 0.5 + fmod(m/2.0,1.0)/m;
            if (x >= n1 && x <= n2) {
                if (x < 0.5)
                    x = fmod(m * x, 1.0) / fmod(m/2.0, 1.0);
                else
                    x = fmod(m * (1.0 - x), 1.0) / fmod(m/2.0, 1.0);
            } else {
                if (x < 0.5)
                    x = fmod(m * x, 1.0);
                else
                    x = fmod(m * (1.0 - x), 1.0);
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 5) {
        printf("Uso: %s <x0> <m> <N> <arquivo_saida>\n", argv[0]);
        return 1;
    }
    double x0 = atof(argv[1]);
    double m = atof(argv[2]);
    size_t N = (size_t)atoll(argv[3]);
    const char *filename = argv[4];

    uint32_t *seq = malloc(N * sizeof(uint32_t));
    if (!seq) {
        fprintf(stderr, "Erro de alocação de memória\n");
        return 1;
    }
    tentmap(x0, m, seq, N);

    FILE *f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Erro ao abrir arquivo para escrita\n");
        free(seq);
        return 1;
    }
    fwrite(seq, sizeof(uint32_t), N, f);
    fclose(f);
    free(seq);
    printf("Arquivo binário gerado: %s\n", filename);
    return 0;
}
