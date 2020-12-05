function main()
    % add CORA to the Matlab path
    addpath(genpath('../code'));

    %center = [0.5; 0.5];
    %generators = [0.5 0; 0 0.5];
    %Z1 = zonotope(center, generators);

    %% setup neural network
    nn = zonoBuNet();
    % specify input range (as intervals or zonotope)
    setInputRange(nn, {[0 1], [0 1]}); % alternatively: setInputRange(nn, Z1);
    % add M and b of conv layer of the form: M * Z + b
    addConvLayer(nn, [1.0 -3.0; 0.0 3.0], [1.0; 1.0]);
    % add relu layer
    addRelu(nn);
    addConvLayer(nn, [1 1.1; -1. 1], [-3; 1.2]);
    % specify robustness criteria: two points have to be specified - they form a line against which the input will be validated
    setRobustness(nn, [1; 1], [2; 2]);

    %% launch verification
    verify(nn, true);

    %% plot results
    axis_scale = [-4 5 0 8];
    plotResults(nn, axis_scale);
end
