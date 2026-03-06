clc;
clear all;
close all;

N = 10*10^6; % total values
init = [0.23];
for i = 1:numel(init)
x = zeros(1,N);


m = 61.81 ; % mu or control parameter
%x0 = transient_effect(init(i),m,200);

x = tentmap(init(i),m,N);



s1 = x(1:N)>.5;
s1 = uint32(x(1:N) .* 2^32);

file = strcat(strcat('rctm',num2str(m)),'.bin');
fid=fopen(file,'w','native');
fwrite(fid,s1,'uint32');
fclose(fid);

end
