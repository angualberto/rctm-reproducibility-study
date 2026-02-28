#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

static inline double mod1(double x) {
    return x - floor(x);
}

/* Núcleo do Robust Chaotic Tent Map */
double rctm_step(double x, double m, double n1, double n2) {

    if (m <= 2.0) {
        // Tent map clássico
        if (x < 0.5)
            return m * x;
        else
            return m * (1.0 - x);
    }

    /* Região interna */
    if (x >= n1 && x <= n2) {
        if (x < 0.5) {
            double num = mod1(m * x);
            double den = mod1(m / 2.0);
            return num / den;  // SEM mod externo (igual ao paper)
        } else {
            double num = mod1(m * (1.0 - x));
            double den = mod1(m / 2.0);
            return num / den;  // SEM mod externo
        }
    }
    /* Região externa */
    else {
        if (x < 0.5)
            return mod1(m * x);
        else
            return mod1(m * (1.0 - x));
    }
}

/* Gerador de bitstream */
void rctm_generate(uint8_t *out, size_t N, double x0, double m) {

    double x = x0;

    double n1 = 0.5 - mod1(m / 2.0) / m;
    double n2 = 0.5 + mod1(m / 2.0) / m;

    for (size_t i = 0; i < N; i++) {
        x = rctm_step(x, m, n1, n2);
        out[i] = (x > 0.5) ? 1 : 0;  // threshold s = 0.5 (Algorithm 2)
    }
}


/* Escrita de PRNG 32-bit para Dieharder/TestU01 */
void write_uint32(const char *fname, double x0, double m, size_t N) {
    FILE *f = fopen(fname, "wb");
    double x = x0;
    double n1 = 0.5 - mod1(m / 2.0) / m;
    double n2 = 0.5 + mod1(m / 2.0) / m;
    for (size_t i = 0; i < N; i++) {
        x = rctm_step(x, m, n1, n2);
        uint32_t val = (uint32_t)(x * 4294967296.0); // 2^32
        fwrite(&val, sizeof(uint32_t), 1, f);
    }
    fclose(f);
}

int main() {

    double x0 = 0.23;
    double m = 61.81;
    size_t N = 5000000; // 5 milhões de uint32_t
    write_uint32("rctm_uint32.bin", x0, m, N);
    printf("Gerado rctm_uint32.bin (%zu números de 32 bits)\n", N);
    return 0;
}
