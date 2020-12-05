classdef  zonoBuNet < handle
    properties
        layers;
        robustness_criteria;
        layerZonos;
        inputRange;
        plot;
    end

    methods
        function NN = zonoBuNet()
            NN.layers = {};
            NN.robustness_criteria = {};
            NN.layerZonos = {};
            NN.inputRange = {};
            NN.plot = {};
        end
        function addConvLayer(NN, M, b)
            NN.layers{end + 1} = {M, b};
        end
        function addRelu(NN)
            NN.layers{end + 1} = {-1};
        end
        %% expects array of intervals with length of n# of dimensions or zonotope representing input intervals
        function setInputRange(NN, input)
            if(iscell(input))
                dimensions = length(input);
                i_center = [];
                i_generators = eye(dimensions);
                for d = 1:dimensions
                    inter = input{2};
                    upperB = inter(2);
                    lowerB = inter(1);
                    i_center = [i_center; (upperB - lowerB) / 2];
                    i_generators(d,:) = i_generators(d,:) * ((upperB - lowerB) / 2);
                end
                NN.inputRange = zonotope(i_center, i_generators);
            elseif(isa(input, 'zonotope'))
                NN.inputRange = input;
            else
                disp("Invalid input for input range!");
            end
        end
        %% expects point of the form p1: [x; y], p2: [x; y]
        function setRobustness(NN, p1, p2)
            r_center = p1;
            r_generators = [(p1(1) - p2(1)) * 10; (p1(2) - p2(2)) * 10];
            NN.robustness_criteria = zonotope(r_center, r_generators);            
        end
        function verify(NN, doConvexHull)
            s = size(NN.layers);
            numberOfLayers = s(2);
            currentZono = NN.inputRange;
            NN.layerZonos{1} = {1, currentZono};
            for i = 1:numberOfLayers
                currentLayer = NN.layers{1, i};
                % relu layer
                if(currentLayer{1} == -1)
                    % here we face a tradeoff between overapproximation and added complexity. in our implementation, we went for
                    % overapproximation, making a convex hull of all the convex hulls we have gathered so far.
                    % alternatively, we open a branch for each convex hull (possible explosion of results with more layers)
                    if(iscell(currentZono))
                        currentZono = convHull(currentZono{:});
                    end
                    [ch, ps] = relu(currentZono, doConvexHull);
                    currentZono = ch;
                    NN.layerZonos{end + 1} = {-1, ps};
                    NN.layerZonos{end + 1} = {-1, ch};
                % convolutional layer
                else
                    % see comment above about overapproximation vs complexity
                    if(iscell(currentZono))
                        currentZono = convHull(currentZono{:});
                    end
                    M = currentLayer{1};
                    b = currentLayer{2};
                    currentZono = conv(currentZono, M, b);
                    NN.layerZonos{end + 1} = {1, currentZono};
                end
            end
            if(iscell(currentZono))
                currentZono = convHull(currentZono{:});
            end
            if(isa(NN.robustness_criteria, 'zonotope'))
                if(isIntersecting(NN.robustness_criteria, currentZono))
                    disp("Input range violates specified robustness criteria!");
                else
                    disp("Robustness criteria are not violated! Successfully verified.");
                end
            end
        end
        % plotting is only supported for 2-dimensional problems
        function plotResults(NN, axis_scale)
            s = size(NN.layerZonos);
            numberOfPlots = s(2);
            plotColumns = numberOfPlots + 1;

            for i = 1:numberOfPlots
                layer = NN.layerZonos{i};
                content = layer{2};
                if(layer{1} == -1)
                    layerS = size(content);
                    layerSize = layerS(1);
                    % it's either a 2x2 (relu_bundles) or a 1x2 (convex hull) matrix
                    if(layerSize == 2)
                        subplot(2, plotColumns, i);
                        hold on;
                        grid on;
                        title('ReLU x');
                        axis(axis_scale);
                        ax = gca;
                        ax.YAxisLocation = 'origin';
                        ax.XAxisLocation = 'origin';
                        axis 'auto y';
                        plotFilled(content{1,2}, [1,2], 'blue', 'FaceAlpha', 0.5);
                        plotFilled(content{1,1}, [1,2], 'blue', 'EdgeColor','red', 'LineWidth', 1);
                        subplot(2, plotColumns, plotColumns + i);
                        hold on;
                        grid on;
                        title('ReLU y');
                        axis(axis_scale);
                        ax = gca;
                        ax.YAxisLocation = 'origin';
                        ax.XAxisLocation = 'origin';
                        axis 'auto y';
                        if(~isempty(content{2,2}))
                            plotFilled(content{2,2}, [1,2], 'blue', 'FaceAlpha', 0.5);
                        end
                        if(~isempty(content{2,1}))
                            plotFilled(content{2,1}, [1,2], 'blue', 'EdgeColor','red', 'LineWidth', 1);
                        end
                    else
                        subplot(2, plotColumns, i);
                        hold on;
                        grid on;
                        title('Convex Hull x');
                        axis(axis_scale);
                        ax = gca;
                        ax.YAxisLocation = 'origin';
                        ax.XAxisLocation = 'origin';
                        axis 'auto y';
                        plotFilled(content{1}, [1,2], 'magenta', 'FaceAlpha', 0.5);
                        xlabel(newline + "   " + newline);
                        subplot(2, plotColumns, plotColumns + i);
                        hold on;
                        grid on;
                        title('Convex Hull y');
                        axis(axis_scale);
                        ax = gca;
                        ax.YAxisLocation = 'origin';
                        ax.XAxisLocation = 'origin';
                        axis 'auto y';
                        plotFilled(content{2}, [1,2], 'magenta', 'FaceAlpha', 0.5);
                        xlabel(newline + "   " + newline);
                    end
                else
                    subplot(2, plotColumns, [i plotColumns + i]);
                    hold on;
                    grid on;
                    axis(axis_scale);
                    ax = gca;
                    ax.YAxisLocation = 'origin';
                    ax.XAxisLocation = 'origin';
                    axis 'auto y';
                    h = subplot(2, plotColumns, [i plotColumns + i]);
                    p = get(gca, 'Position');
                    p(2) = 3 * p(2);
                    p(4) = p(4) * 0.5;
                    set(gca, 'Position', p);
                    xlabel(newline + "   " + newline);
                    if(i == 1)
                        title('Input z1');
                    elseif(i == numberOfPlots)
                        title(['Output z' int2str(i)]);
                        if(isa(NN.robustness_criteria, 'zonotope'))
                            plot(NN.robustness_criteria);
                            isIntersecting(NN.robustness_criteria, content);
                        end
                    else
                        title(['Zonotope z' int2str(i)]);
                    end
                    plotFilled(content, [1,2], 'green', 'FaceAlpha',0.5);      
                end
            end
        saveas(gcf, '../results/verification_plot.png');
        end
    end

end