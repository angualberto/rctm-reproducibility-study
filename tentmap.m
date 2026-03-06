function x = tentmap(x0,m,N)

x = zeros(1,N);

x(1) = x0;

n1 = .5 - mod(m/2,1)/m;
n2 = .5 + mod(m/2,1)/m;

e_p = 1;
for n = 1:N-1
    
    
    if m <=2
        if  x(n) <.5
            x(n+1) = m*x(n) *e_p;
        else
            x(n+1) = m*(1 - x(n) )*e_p;
        end
    else
         if x(n)>= n1 && x(n)<=n2
            if x(n) <.5
                x(n+1) = mod(m*x(n)*e_p,1)/mod(m/2,1);
            else
                x(n+1) = mod(m*(1 - x(n))*e_p,1)/mod(m/2,1);
            end
         else
             if x(n) <.5
                x(n+1) = mod(m*x(n)*e_p,1);
            else
                x(n+1) = mod(m*(1 - x(n))*e_p,1);
            end
        end
    end
end
end