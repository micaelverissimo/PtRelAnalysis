% Neural Network Training Process
% author: Natanael Junior (natmourajr@gmail.com)
% LPS - Signal Processing Lab.
% UFRJ - Brazil

% Steps
% 1 - Data Aquisition
% 2 - Normalization (data, targets)
% 3 - Split Training Sets (train, test, validation)
% 4 - Training Process
% 5 - Result Analysis
% 
% In machine learning and related fields, artificial neural networks (ANNs)
% are computational models inspired by an animal's central nervous systems 
% (in particular the brain), and are used to estimate or approximate functions 
% that can depend on a large number of inputs and are generally unknown. 
% Artificial neural networks are generally presented as systems of interconnected 
% "neurons" which can compute values from inputs, and are capable of machine 
% learning as well as pattern recognition thanks to their adaptive nature.
% 

close all; % close all figure windows
clear all; % clear all variables
clc; % reset command line

fprintf('Starting %s.m\n',mfilename('fullpath'));
fprintf('Importing Functions\n');
addpath(genpath('functions'));

% 1 - Data Aquisition
input_file_name = '../Results/TxtFiles/TxtFile_lvbb125.txt';

fid = fopen(input_file_name, 'r');

C = textscan(fid, '%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f');

[inputs1 inputs2 inputs3 inputs4 inputs5 inputs6 inputs7 inputs8 inputs9 inputs10 inputs11 inputs12 inputs13 inputs14 inputs15 inputs16 inputs17 inputs18 inputs19 inputs20 inputs21 inputs22 inputs23 inputs24 inputs25 inputs26 inputs27 inputs28 inputs29 inputs30 inputs31 inputs32 inputs33 inputs34 inputs35 inputs36 inputs37]= deal(C{:});

inputs = [inputs1 inputs2 inputs3 inputs4 inputs5 inputs6 inputs7 inputs8 inputs9 inputs10 inputs11 inputs12 inputs13 inputs14 inputs15 inputs16 inputs17 inputs18 inputs19 inputs20 inputs21 inputs22 inputs23 inputs24 inputs25 inputs26 inputs27 inputs28 inputs29 inputs30 inputs31 inputs32 inputs33 inputs34 inputs35 inputs36 inputs37];
inputs_label = {'Pt_EM', 'Eta_EM', 'Phi_EM', 'E_EM', 'PtRel_EM', 'FracEM3', 'FracTile0', 'TrkWidth', 'EMF', 'JVF', 'NTrk', 'SumPtTrk', 'Pt_EMJES', 'Eta_EMJES', 'Phi_EMJES', 'E_EMJES', 'PtRel_EMJES', 'Pt_EMJESGSC', 'Eta_EMJESGSC', 'Phi_EMJESGSC', 'E_EMJESGSC', 'PtRel_EMJESGSC', 'Pt_EMJESGSCMu', 'Eta_EMJESGSCMu', 'Phi_EMJESGSCMu', 'E_EMJESGSCMu', 'PtRel_EMJESGSCMu', 'Pt_EMJESGSCMuPt', 'Eta_EMJESGSCMuPt', 'Phi_EMJESGSCMuPt', 'E_EMJESGSCMuPt', 'PtRel_EMJESGSCMuPt', 'Pt_GENWZ', 'Eta_GENWZ', 'Phi_GENWZ', 'E_GENWZ', 'PtRel_GENWZ'};
    %'RecoPt','RecoE','RecoEta','RecoPhi','MuEffectPt', 'NTrk','SumPtTrk','JVF','SvpLxy','Width','EMF','TrkWidth','BWeight'};
clear('fid','C','input_file_name','inputs1','inputs2','inputs3','inputs4','inputs5','inputs6','inputs7','inputs8','inputs9','inputs10','inputs11', 'inputs12', 'inputs13', 'inputs14', 'inputs15', 'inputs16', 'inputs17', 'inputs18', 'inputs19', 'inputs20', 'inputs21', 'inputs22', 'inputs23', 'inputs24', 'inputs25', 'inputs26', 'inputs27', 'inputs28', 'inputs29', 'inputs30', 'inputs31', 'inputs32', 'inputs33', 'inputs34', 'inputs35' , 'inputs36', 'inputs37');


select_inputs = [6 7 8 9 10 11 12 23 24 25 26 27 33 34 35 36 37]; % all
%select_inputs = 1:size(inputs,2); % reco_pt


inputs = inputs(:,select_inputs);
inputs_label = inputs_label(select_inputs);

% analyzing Autocorrelation
 figure;
 imagesc(corr(inputs));
 colormap(1-gray); colorbar;
 title('AutoCorrelation Matrix','FontSize', 15,'FontWeight', 'bold');
 set(gca,'XTick',1:size(inputs,2));
 set(gca,'YTick',1:size(inputs,2));
 set(gca,'XTickLabel',inputs_label);
 set(gca,'YTickLabel',inputs_label);
 %rotateXLabels(gca,90); 
% fig2pdf(gcf,'autocorr.pdf');
% saveas(gcf,'autocorr.jpg');
% close(gcf);

% 3 - Split Training Sets (train, test, validation)

n_tests = 10;
CVO = cvpartition(size(inputs,1),'Kfold',n_tests); % split into n_tests tests
for i = 1:n_tests
    fprintf('Split Training Sets\n');
    
    trn_id =  CVO.training(i); % taking the first one
    tst_id =  CVO.test(i); % taking the first one
    val_id = tst_id; % test = validation -> small statistics


    % turn trn_id, tst_id in integers  to use in NN training process
    itrn = [];
    itst = [];
    ival = [];

    for i = 1:length(trn_id)
        if trn_id(i) == 1
            itrn = [itrn; i];
        else 
            itst = [itst; i];
        end
    end
    ival = itst;

    % 2 - Using train set to extract normalization factors
    fprintf('Normalizing Inputs\n');


    inputs_norm = [];
    ps = [];
    truth_pt = inputs(:,end);
    [a,b] = hist(truth_pt(itrn,:),100);
    peak_truth_pt = b(find(a==max(a)));
    if true
        % mean = 0, var = 1
        [~,ps] = mapstd(inputs(itrn,:)'); % ps - normalization factors
        % applying normalization in all events
        % mapstd -> mean = 0, std = 1;
        inputs_norm =  mapstd('apply',inputs',ps)';
    else
        % reco_pt/truth_pt
        ps.xoffset(1) = 0; 
        ps.gain(1) = 1/peak_truth_pt;
        inputs_norm = ps.gain(1)*inputs; 
    end

    % Output
    fprintf('Normalizing Outputs\n');
    ps = [];
    truth_pt = inputs(:,end);
    [a,b] = hist(truth_pt(itrn,:),100);
    peak_truth_pt = b(find(a==max(a)));
    if true
        % truth_pt/reco_pt
        truth_pt_norm=truth_pt./inputs(:,8);
    else
        % truth_pt/mop(truth_pt)
        truth_pt_norm=truth_pt./peak_truth_pt;
    end

    nn_inputs=inputs_norm(:,1:end-1);
    nn_target=truth_pt_norm;


    % 4 - Training Process
    fprintf('Training Process\n');

    top = 10; % number of neurons in hidden layer
    train_fnc = 'trainlm'; % weights update function
    perf_fnc = 'mse'; % error function % without regularization
    %perf_fnc = 'msereg'; % error function % with regularization
    act_fnc = {'tansig' 'purelin'}; % activation function
    n_epochs = 100;

    show = true;

    [trained_nn, train_description] = train_neural_network(nn_inputs', nn_target', itrn, ival, itst, top, train_fnc, perf_fnc, act_fnc, n_epochs, show);

    nn_output = sim(trained_nn, inputs_norm');

end
% 5 - Result Analysis
fprintf('Result Analysis\n');

% train analysis
plotperform(train_description);
fig2pdf(gcf,'training_description.pdf');
close(gcf);


hist((targets'-nn_output),100);
title('Error Histogram','FontSize', 15,'FontWeight', 'bold');
xlabel('Error Values','FontSize', 15,'FontWeight', 'bold');
fig2pdf(gcf,'error_hist.pdf');
close(gcf);
return

%input_labels = {sprintf('((reco_pt-%1.6f)*%1.6f)', ps.xoffset(1), ps.gain(1))};
% export a formula
fprintf('Export Formula\n');
Formula_labels = {};
for i = 1:length(select_inputs)
    Formula_labels{i} = sprintf('((%s-%1.6f)*%1.6f)', inputs_label{i}, ps.xoffset(i), ps.gain(i));
end

fprintf('Out: %s\n',GetNNFormula(trained_nn,'TMath::TanH',Formula_labels));
%fprintf('Out: %s\n',GetNNFormula(trained_nn,{'reco_norm'}));

nn_resul = sim(trained_nn, inputs_norm(1));

y1 = tansig(trained_nn.IW{1,:}*inputs_norm(1,:)' + trained_nn.b{1});
m_resul = trained_nn.LW{2,:}*y1+ trained_nn.b{2};

diff_resul = nn_resul - m_resul;
fprintf('diff between results: %1.6f\n',diff_resul);


fprintf('Exporting Functions\n');
rmpath(genpath('functions'));

fprintf('THE END!!!\n');


