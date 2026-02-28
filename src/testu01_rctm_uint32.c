#include <stdio.h>
#include <stdint.h>
#include "unif01.h"
#include "bbattery.h"

#define NWORDS 5000000
static FILE *f = NULL;
static uint32_t buffer[NWORDS];
static size_t idx = 0, loaded = 0;

unsigned long rctm32(void) {
    if (idx >= loaded) {
        loaded = fread(buffer, sizeof(uint32_t), NWORDS, f);
        idx = 0;
        if (loaded == 0) return 0;
    }
    return buffer[idx++];
}

int main() {
    f = fopen("/home/andre/Documentos/rctm_uint32.bin", "rb");
    if (!f) {
        printf("Arquivo rctm_uint32.bin não encontrado!\n");
        return 1;
    }
    unif01_Gen *gen = unif01_CreateExternGenBits32("RCTM32", rctm32);
    bbattery_SmallCrush(gen);
    unif01_DeleteExternGenBits32(gen);
    fclose(f);
    return 0;
}
