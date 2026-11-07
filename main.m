clear all;
close all;
clc;
addpath(genpath('.'));

%% Load the data set
load 3Sources.mat;
gt = y;
fea = X;

classid = gt;
cls_num = length(unique(classid));
inPara.cluster = cls_num;
for v = 1:length(fea)
    X{v} = NormalizeFea(fea{v}',0);
    [d(v),~] = size(X{v});
end
k = length(unique(gt));  

%% Parameter setting
disp('Performing ...');

para1_list = [cls_num, 2 * cls_num, 3 * cls_num];    % anchor number 
para2_list = [1e3, 1e2, 1e1, 1, 1e-1, 1e-2, 1e-3, 1e-4]; % lam1
para3_list = [1e3, 1e2, 1e1, 1, 1e-1, 1e-2, 1e-3, 1e-4]; % lam2
para4_list = [1, 2, 3, 4, 5]; % 层次

mu = 0.001; 
mu2 = 0.001;

total_runs = length(para1_list) * length(para2_list) * length(para3_list) * length(para4_list);
fprintf('Total parameter combinations: %d\n', total_runs);

results_table = table();

run_counter = 0;

timestamp = datestr(now, 'yyyymmdd_HHMMSS');
excel_filename = sprintf('parameter_sweep_results_%s.xlsx', timestamp);
mat_filename = sprintf('parameter_sweep_results_%s.mat', timestamp);
fprintf('Results will be saved to: %s and %s\n', excel_filename, mat_filename);

for para1 = para1_list
    for para2 = para2_list 
        for para3 = para3_list
            for para4 = para4_list
                
                run_counter = run_counter + 1;
                fprintf('\n--- Running parameter set %d/%d ---\n', run_counter, total_runs);
                fprintf('para1=%.0e, para2=%d, para3=%.0e, para4=%.0e', ...
                        para1, para2, para3, para4);
                
                inPara.para1 = para1; 
                inPara.para2 = para2; 
                inPara.para3 = para3;
                inPara.para4 = para4;

                
                inPara.max_iter = 100;
                inPara.mu = 0.001;
                inPara.mu2 = 0.001;
                
                ii = 10;
                
                ACC = zeros(1, ii);
                NMI = zeros(1, ii);
                Purity = zeros(1, ii);
                Fscore = zeros(1, ii);
                Precision = zeros(1, ii);
                Recall = zeros(1, ii);
                AR = zeros(1, ii);
                Entropy = zeros(1, ii);
                run_times = zeros(1, ii);


                start_time = tic;
                    
                [outB, outZ, outP, outA] = LMVSCHAPTAM(X, inPara, k);
                
                for i = 1:ii                 

                    run_times(i) = toc(start_time);
                    
                    %% Construct the consistent bipartite graph and obtain clustering results
                    result_idx=litekmeans(U, k, 'MaxIter', 100,'Replicates',10);
                    metrics = Clustering8Measure(classid, result_idx);

                    ACC(i) = metrics(1);
                    NMI(i) = metrics(2);
                    Purity(i) = metrics(3);
                    Fscore(i) = metrics(4);
                    Precision(i) = metrics(5);
                    Recall(i) = metrics(6);
                    AR(i) = metrics(7);
                    Entropy(i) = metrics(8);
                    
                    fprintf('  Iteration %d: ACC=%.4f, NMI=%.4f, Purity=%.4f, Fscore=%.4f, Time=%.2f sec\n', ...
                            i, ACC(i), NMI(i), Purity(i), Fscore(i), run_times(i));
                end

                ACC_mean = mean(ACC);
                ACC_std = std(ACC);
                NMI_mean = mean(NMI);
                NMI_std = std(NMI);
                Purity_mean = mean(Purity);
                Purity_std = std(Purity);
                Fscore_mean = mean(Fscore);
                Fscore_std = std(Fscore);
                Precision_mean = mean(Precision);
                Precision_std = std(Precision);
                Recall_mean = mean(Recall);
                Recall_std = std(Recall);
                AR_mean = mean(AR);
                AR_std = std(AR);
                Entropy_mean = mean(Entropy);
                Entropy_std = std(Entropy);
                
                avg_time = mean(run_times);
                total_time = sum(run_times);
                
                fprintf('Results: ACC=%.4f±%.4f, NMI=%.4f±%.4f, Purity=%.4f±%.4f, Fscore=%.4f±%.4f, Avg Time=%.2f sec\n', ...
                        ACC_mean, ACC_std, NMI_mean, NMI_std, Purity_mean, Purity_std, Fscore_mean, Fscore_std, avg_time);
                
                new_row = table(run_counter, para1, para2, para3, para4, ...
                               ACC_mean, ACC_std, NMI_mean, NMI_std, ...
                               Purity_mean, Purity_std, Fscore_mean, Fscore_std, ...
                               Precision_mean, Precision_std, Recall_mean, Recall_std, ...
                               AR_mean, AR_std, Entropy_mean, Entropy_std, ...
                               avg_time, total_time, ii, ...
                               'VariableNames', {'run_id', 'para1', 'para2', 'para3', 'para4', ...
                                                 'ACC_mean', 'ACC_std', 'NMI_mean', 'NMI_std', ...
                                                 'Purity_mean', 'Purity_std', 'Fscore_mean', 'Fscore_std', ...
                                                 'Precision_mean', 'Precision_std', 'Recall_mean', 'Recall_std', ...
                                                 'AR_mean', 'AR_std', 'Entropy_mean', 'Entropy_std', ...
                                                 'Avg_Time_sec', 'Total_Time_sec', 'Num_Iterations'});
                results_table = [results_table; new_row];
                
                save(mat_filename, 'results_table');
                writetable(results_table, excel_filename);
                
                fprintf('Table size: %d rows\n', height(results_table));
                
            end
        end
    end
end

disp(' ');
disp('========================================');
disp('Parameter sweep completed!');
disp('========================================');
disp('Summary of results:');
disp(results_table);

[best_ACC, best_idx_ACC] = max(results_table.ACC_mean);
[best_NMI, best_idx_NMI] = max(results_table.NMI_mean);
[best_Purity, best_idx_Purity] = max(results_table.Purity_mean);
[best_Fscore, best_idx_Fscore] = max(results_table.Fscore_mean);

fprintf('\n=== Best Results by Different Metrics ===\n');

fprintf('\nBest ACC: %.4f (std=%.4f)\n', best_ACC, results_table.ACC_std(best_idx_ACC));
fprintf('  para1=%.0e, para2=%d, para3=%.0e, para4=%.0e\n', ...
        results_table.para1(best_idx_ACC), results_table.para2(best_idx_ACC), ...
        results_table.para3(best_idx_ACC), results_table.para4(best_idx_ACC));

fprintf('\nBest NMI: %.4f (std=%.4f)\n', best_NMI, results_table.NMI_std(best_idx_NMI));
fprintf('  para1=%.0e, para2=%d, para3=%.0e, para4=%.0e\n', ...
        results_table.para1(best_idx_NMI), results_table.para2(best_idx_NMI), ...
        results_table.para3(best_idx_NMI), results_table.para4(best_idx_NMI));

fprintf('\nBest Purity: %.4f (std=%.4f)\n', best_Purity, results_table.Purity_std(best_idx_Purity));
fprintf('  para1=%.0e, para2=%d, para3=%.0e, para4=%.0e\n', ...
        results_table.para1(best_idx_Purity), results_table.para2(best_idx_Purity), ...
        results_table.para3(best_idx_Purity), results_table.para4(best_idx_Purity));

fprintf('\nBest Fscore: %.4f (std=%.4f)\n', best_Fscore, results_table.Fscore_std(best_idx_Fscore));
fprintf('  para1=%.0e, para2=%d, para3=%.0e, para4=%.0e\n', ...
        results_table.para1(best_idx_Fscore), results_table.para2(best_idx_Fscore), ...
        results_table.para3(best_idx_Fscore), results_table.para4(best_idx_Fscore));

save(mat_filename, 'results_table');
writetable(results_table, excel_filename);
fprintf('\nFinal results saved to:\n');
fprintf('  Excel file: %s\n', excel_filename);
fprintf('  MAT file: %s\n', mat_filename);
