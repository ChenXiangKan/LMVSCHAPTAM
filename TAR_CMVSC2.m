function [outJ, outZ, outP, outA] = TAR_CMVSC2(inData, inPara, k)

    inX = inData;
    numViews = length(inX);
    [~, numSamples] = size(inX{1});
    numAnchors = inPara. para1;
    lam = inPara. para2;
    lam2 = inPara. para3;
    U = inPara. para4;

    P_all = [];
    m = 0;
    Y = [];

    %% Initialize demension
    l_dims = zeros(1, U+1);
    l_dims(1) = numAnchors;
    l_dims(2) = numAnchors;
    if U == 2
        l_dims(3) = numAnchors + k;
    end

    if U >= 3
        for u = 3:U+1
            l_dims(u) = l_dims(u-1) + k;
        end
    end
    projnum = l_dims(U+1);

    for v = 1:numViews
        [P_all{v}] = pca(inX{v}');
        XH{v} = inX{v};
        T{v} = P_all{v}(:, 1:numAnchors);
        XHT{v} = inX{v}';
        A{v} = zeros(numAnchors,numAnchors);      
        J{v} = zeros(numAnchors,numAnchors);
        m = m + size(inX{v}, 1);   
        Y = vertcat(Y, inX{v});
    end
    J{numViews+1} = zeros(numAnchors,numAnchors);
    for v = 1:numViews+1
        Y1{v} = zeros(numAnchors,numAnchors);
    end

    C = zeros(projnum, numSamples);
    P = zeros(m , projnum);
    Z = zeros(numAnchors, numSamples);
    Z(1:numAnchors, 1:numAnchors) = eye(numAnchors);


    

    D_cell = cell(1, U);

    for w = 1:U
        rows = l_dims(w+1);
        cols = l_dims(w);
        D_cell{w} = zeros(rows, cols);
    end

    max_iter = inPara. max_iter;
    mu = inPara. mu;

    max_mu = 10e6;

    eta_mu = 1.8;

    tol = 1*10^ - 6;
    iter = 0;
    flag = 1;

    obj = [];
    err2_history = [];
    iter_record = [];

    while flag
        iter = iter + 1;



        %% Update P
        temp_P = Y * C';
        [P_u, ~, P_v] = svd(temp_P, 'econ');
        P = P_u * P_v';


        %% Update D
        if U == 1
            M = C * Z' - 0.5 * Y1{numViews+1} + mu / 2 * J{numViews+1};
            [U_M, ~, V_M] = svd(M, 'econ');
            D_cell{1} = U_M * V_M';
            H = D_cell{1};
        end
        if U >= 2
            for w = 1:U
                D_cell{w} = update_Dw(D_cell, C, Z, numAnchors, U, projnum, numViews, w, Y1, J, mu);
            end
            H = D_cell{1}; %Du*Du-1...*D1
            for u = 2:U
                H = D_cell{u} * H;
            end
        end

        %% Update C
        C = 0.5*(H * Z + P' * Y);


        %% Update T
        for v = 1:numViews
            temp_T{v} = XH{v} * (Z' * A{v}');
            [T_u, ~, T_v] = svd(temp_T{v}, 'econ');
            T{v} = T_u * T_v';
        end

        %% Update Ai
        for v = 1:numViews
            temp_A{v} = T{v}' * XH{v} * Z' - Y1{v} / 2 + mu * J{v} / 2;
            A{v} = temp_A{v}*pinv (Z * Z' + (mu / 2) * eye(numAnchors));
        end



        %% Update Z
        N = 0;
        AA = 0;
        for v = 1:numViews
            N = N + (T{v} * A{v})' * XH{v};
            AA = AA + A{v}' * A{v};
        end
        Z = ((1 + lam) * eye(numAnchors) + AA) \ (H' * C + N); 
        Z = SimplexProj(Z');
        Z = Z';


        %% Update J
        A_tensor = cat(3, A{:, :},D_cell{1});
        Y1_tensor = cat(3, Y1{:, :});
        bvec_A = A_tensor(:);
        bvec_Y = Y1_tensor(:);
        sX = size(A_tensor);

        [bvec_J, ~] = wshrinkObj(bvec_A + (1 / mu) * bvec_Y, lam2 / mu, sX, 0, 1);
        J_tensor = reshape(bvec_J, sX);



        %% Update Y
        Y1_tensor = A_tensor-J_tensor;
        for v = 1:numViews+1
            Y1{v} = Y1{v} + mu*Y1_tensor(:,:,v);
        end

        %% Update mu
        mu = min(eta_mu * mu, max_mu);



        diff1(iter) = norm(abs(A_tensor(:) - J_tensor(:)), 'fro');
        err1 = diff1(iter);


        obj = [obj; err1];
        iter_record = [iter_record; iter];
        
%         save('res\3Sources.mat', 'obj');


        if (err1 < tol) || iter >= max_iter
            flag = 0;
            outJ = J;
            outZ = Z;
            outP = P;
            outA = A;

            clear J;
            clear Z;
            clear P;
            clear A;
            clear numViews;
        end

    end
fprintf('==%d==',iter);

end