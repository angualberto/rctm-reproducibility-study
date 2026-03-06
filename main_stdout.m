% Configurações para 500MB
N = 131072000; 
init = 0.23;
m = 61.81;

% Gera os dados (certifique-se que tentmap.m está na mesma pasta)
x = tentmap(init, m, N);

% Converte para uint32 (4 bytes cada)
s1 = uint32(x .* 2^32);

% Limpa x para liberar RAM (seu MS-7C96 agradece)
clear x;

% Envia para o stdout (Pipe)
fwrite(stdout, s1, 'uint32');
fflush(stdout);
