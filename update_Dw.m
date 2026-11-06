function D_w_new = update_Dw(D_cell, C, Z, numAnchors, U, projnum, numViews, w, Y1, J, mu)


    if U == 2
        if w == 1
            M = D_cell{2}' * C * Z' - 0.5 * Y1{numViews+1} + mu / 2 * J{numViews+1};
            [U_F, ~, V_F] = svd(M, 'econ');
            D_w_new = U_F * V_F';  
        end
        if w == 2
            F = C * Z' * D_cell{1}';
            [U_F, ~, V_F] = svd(F, 'econ');
            D_w_new = U_F * V_F';  
        end
    end


    if U >= 3
        if w == U
            K = eye(projnum);  
        else
            K = D_cell{U};
            for u = U-1:-1:w+1
                K = K * D_cell{u};
            end
        end

        if w == 1
            R = eye(numAnchors);
        else
            R = D_cell{w-1};
            for u = w-2:-1:1
                R = R * D_cell{u};
            end
        end

        if w == 1
            M = K' * C * Z' - 0.5 * Y1{numViews+1} + mu / 2 * J{numViews+1};
            [U_F, ~, V_F] = svd(M, 'econ');
            D_w_new = U_F * V_F';  
        else

        F = K' * C * Z' * R';
        [U_F, ~, V_F] = svd(F, 'econ');
        D_w_new = U_F * V_F';
        end

    end
end