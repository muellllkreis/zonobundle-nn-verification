function [ch, ps] = relu(input, doConvexHull)
    if isa(input,'zonotope') | isa(input, 'zonoBundle')
        verts = input.vertices;

        dimensions = length(verts(:,1));
        dim_names = ['x', 'y', 'z'];
        min_max = zeros(dimensions, 2);

        for d = 1:dimensions
            min_max(d, 1) = min(verts(d,:));
            min_max(d, 2) = max(verts(d,:));
        end

        halfspaces{dimensions, 2} = [];
        relu_bundles{dimensions, 2} = [];

        g_gen = eye(dimensions);
        l_gen = eye(dimensions);

        for d = 1:dimensions
            %fprintf("Iteration %i (halfspaces for %c)\n", d, dim_names(d));
            V = min_max(d,:);
            if(~any(diff(sign(V(V~=0)))))
                %fprintf("All min_max have same sign for dimension %c\n", dim_names(d));
                if(max(V)>0)
                    halfspaces{d, 2} = input;
                    relu_bundles{d, 2} = zonoBundle({input, halfspaces{d, 2}});
                else
                    halfspaces{d, 1} = input;
                    relu_bundles{d, 1} = zonoBundle({input, halfspaces{d, 1}});
                end
                break;
            end
            % for each dimension, prepare two zonotopes:
            % 1) dimension > 0 - replace min_dimension-value (d,1) with 0, rest stays the same
            % 2) dimension < 0 - replace max_dimension-value (d,2) with 0, rest stays the same
            % then make zonotopes like:
            % gx_center = [gx_max_x / 2; (gx_max_y - abs(gx_min_y))/2 + gx_min_y];
            % gx_gen = [abs(gx_max_x)/2 0; 0 (gx_max_y - abs(gx_min_y))/2];
            % then add to halfspaces{d,1} and halfspaces{d,2} respectively
            g_center = [];
            l_center = [];
            for e = 1:dimensions
                if(e == d)
                    g_center = [g_center; min_max(e, 2) / 2];
                    g_gen(e,:) = g_gen(e,:) * abs(min_max(e, 2)/2);

                    % same for l_center, l_gen...
                    l_center = [l_center; min_max(e, 1) / 2];
                    l_gen(e,:) = l_gen(e,:) * abs(min_max(e, 1)/2);
                else
                    g_center = [g_center; ((min_max(e, 2) - abs(min_max(e, 1))) / 2) + min_max(e, 1)];
                    g_gen(e,:) = g_gen(e,:) * ((min_max(e, 2) - abs(min_max(e, 1)))/2);

                    l_center = [l_center; ((min_max(e, 2) - abs(min_max(e, 1))) / 2) + min_max(e, 1)];
                    l_gen(e,:) = l_gen(e,:) * ((min_max(e, 2) - abs(min_max(e, 1)))/2);
                end
            end
            halfspaces{d, 1} = zonotope(l_center, l_gen);
            halfspaces{d, 2} = zonotope(g_center, g_gen);

            relu_bundles{d, 1} = zonoBundle({input, halfspaces{d, 1}});
            relu_bundles{d, 2} = zonoBundle({input, halfspaces{d, 2}});
        end


        l_gen = eye(dimensions);
        bias = zeros(dimensions,dimensions);
        % Create (d-1)-dimensional representation for l-halfspace (apply ReLu)
        for d = 1:dimensions
            if(~isempty(relu_bundles{d, 1}))
                zb = relu_bundles{d, 1};
            else
                break;
            end
            verts = zb.vertices;
            for f = 1:dimensions
                min_max(f, 1) = min(verts(f,:));
                min_max(f, 2) = max(verts(f,:));
            end
            l_center = [];
            for e = 1:dimensions
                if(e == d)
                    l_center = [l_center; 0];
                    l_gen(e, :) = 0;
                    bias(e, e) = 0;
                else
                    l_center = [l_center; ((min_max(e, 2) - abs(min_max(e, 1))) / 2) + min_max(e, 1)];
                    l_gen(e, :) = l_gen(e, :) * ((min_max(e, 2) - abs(min_max(e, 1)))/2);
                    bias(e, d) = -((min_max(e, 2) - abs(min_max(e, 1))));
                end
            end
            relu_bundles{d, 1} = zonotope(l_center, l_gen);
        end

        % Sum zono bundles for each dimension 
        if(doConvexHull)
            convex_hulls{dimensions} = [];
            for d = 1:dimensions
                convex_hulls{d} = bias(:,d);
                if(~isempty(relu_bundles{d, 1}))
                    convex_hulls{d} = convex_hulls{d} + relu_bundles{d,1};
                end
                if(~isempty(relu_bundles{d, 2}))
                    convex_hulls{d} = convex_hulls{d} + relu_bundles{d,2};
                end
            end
            ch = convex_hulls;
            ps = relu_bundles;
        else
            ch = relu_bundles;
            ps = relu_bundles;
        end
    else
        disp("Invalid input");
        ch = {};
        ps = {};
    end
end