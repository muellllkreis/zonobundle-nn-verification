function z = conv(input, M, b)
    if isa(input,'zonotope') | isa(input, 'zonoBundle')
        z = (M * input) + b;
    else
        display("Bad input!");
        z = null;
    end
end