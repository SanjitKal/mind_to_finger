% Pipeline_real.m - runs the predicion pipeline for leaderboard predictions:
% 1. extracts data and splits into training and testing
% 2. filters the data
% 3. computes features
% 4. creates the R matrix for linear regression
% 5. trains linear regrssion model and computes predicitons
%% Extract dataglove and ECoG data 
% Dataglove should be (samples x 5) array 
% ECoG should be (samples x channels) array
glove_1 = train_dg(1);
glove_2 = train_dg(2);
glove_3 = train_dg(3);


ecog_1 = train_ecog(1);
ecog_2 = train_ecog(2);
ecog_3 = train_ecog(3);

lead_ecog_1 = leaderboard_ecog(1);
lead_ecog_2 = leaderboard_ecog(2);
lead_ecog_3 = leaderboard_ecog(3);

num_ch_1 = 61;
num_ch_2 = 46;
num_ch_3 = 64;

% Split data into a train and test set (use at least 50% for training)
num_train = 200000;
num_test = 100000;

glove_1_train = glove_1{1}(:,1:5);
ecog_1_train = ecog_1{1}(:, 1:num_ch_1);
ecog_1_test = lead_ecog_1{1}(:, 1:num_ch_1);

glove_2_train = glove_2{1}(:,1:5);
ecog_2_train = ecog_2{1}(:, 1:num_ch_2);
ecog_2_test = lead_ecog_2{1}(:, 1:num_ch_2);

glove_3_train = glove_3{1}(:,1:5);
ecog_3_train = ecog_3{1}(:, 1:num_ch_3);
ecog_3_test = lead_ecog_3{1}(:, 1:num_ch_3);
%% Get Features
% run getWindowedFeats_release function
% test_feats = getWindowedFeats(ecog_1_train(1:1000,:), 1000, .1, .05);
ecog_1_train_feats = getWindowedFeats(ecog_1_train, 1000, .1, .05);
%%
ecog_2_train_feats = getWindowedFeats(ecog_2_train, 1000, .1, .05);
%%
ecog_3_train_feats = getWindowedFeats(ecog_3_train, 1000, .1, .05);
%%
ecog_1_test_feats = getWindowedFeats(ecog_1_test, 1000, .1, .05);
%%
ecog_2_test_feats = getWindowedFeats(ecog_2_test, 1000, .1, .05);
%%
ecog_3_test_feats = getWindowedFeats(ecog_3_test, 1000, .1, .05);


%% Create R matrix
ecog_1_train_R = create_R_matrix(ecog_1_train_feats, 3);
ecog_2_train_R = create_R_matrix(ecog_2_train_feats, 3);
ecog_3_train_R = create_R_matrix(ecog_3_train_feats, 3);
%%
ecog_1_test_R = create_R_matrix(ecog_1_test_feats, 3);
ecog_2_test_R = create_R_matrix(ecog_2_test_feats, 3);
ecog_3_test_R = create_R_matrix(ecog_3_test_feats, 3);
%% Train classifiers (8 points)
% Classifier 1: Get angle predictions using optimal linear decoding. That is, 
% calculate the linear filter (i.e. the weights matrix) as defined by 
% Equation 1 for all 5 finger angles.

% downsample the gloves

chunk_sz = floor(length(glove_1_train(:,1))/length(ecog_1_train_R(:,1)));
num_chunks = length(glove_1_train(:,1)) / chunk_sz;

glove_1_train_ds = zeros(num_chunks, length(glove_1_train(1,:)));
glove_2_train_ds = zeros(num_chunks, length(glove_2_train(1,:)));
glove_3_train_ds = zeros(num_chunks, length(glove_3_train(1,:)));

for i = 1:5
    glove_1_train_ds(:,i) = decimate(glove_1_train(:,i), chunk_sz);
    glove_2_train_ds(:,i) = decimate(glove_2_train(:,i), chunk_sz);
    glove_3_train_ds(:,i) = decimate(glove_3_train(:,i), chunk_sz);
end
%% Extend the R matrices by one row to match downsampled dataglove data size
ecog_1_train_R_ext = [ecog_1_train_R; ecog_1_train_R(length(ecog_1_train_R(:,1)),:)];
ecog_2_train_R_ext = [ecog_2_train_R; ecog_2_train_R(length(ecog_2_train_R(:,1)),:)];
ecog_3_train_R_ext = [ecog_3_train_R; ecog_3_train_R(length(ecog_3_train_R(:,1)),:)];
%% Create the filters for linear regression
%create filters
f11 = mldivide(ecog_1_train_R_ext.'*ecog_1_train_R_ext,ecog_1_train_R_ext.'*glove_1_train_ds(:,1));
f12 = mldivide(ecog_1_train_R_ext.'*ecog_1_train_R_ext,ecog_1_train_R_ext.'*glove_1_train_ds(:,2));
f13 = mldivide(ecog_1_train_R_ext.'*ecog_1_train_R_ext,ecog_1_train_R_ext.'*glove_1_train_ds(:,3));
f14 = mldivide(ecog_1_train_R_ext.'*ecog_1_train_R_ext,ecog_1_train_R_ext.'*glove_1_train_ds(:,4));
f15 = mldivide(ecog_1_train_R_ext.'*ecog_1_train_R_ext,ecog_1_train_R_ext.'*glove_1_train_ds(:,5));

f21 = mldivide(ecog_2_train_R_ext.'*ecog_2_train_R_ext,ecog_2_train_R_ext.'*glove_2_train_ds(:,1));
f22 = mldivide(ecog_2_train_R_ext.'*ecog_2_train_R_ext,ecog_2_train_R_ext.'*glove_2_train_ds(:,2));
f23 = mldivide(ecog_2_train_R_ext.'*ecog_2_train_R_ext,ecog_2_train_R_ext.'*glove_2_train_ds(:,3));
f24 = mldivide(ecog_2_train_R_ext.'*ecog_2_train_R_ext,ecog_2_train_R_ext.'*glove_2_train_ds(:,4));
f25 = mldivide(ecog_2_train_R_ext.'*ecog_2_train_R_ext,ecog_2_train_R_ext.'*glove_2_train_ds(:,5));

f31 = mldivide(ecog_3_train_R_ext.'*ecog_3_train_R_ext,ecog_3_train_R_ext.'*glove_3_train_ds(:,1));
f32 = mldivide(ecog_3_train_R_ext.'*ecog_3_train_R_ext,ecog_3_train_R_ext.'*glove_3_train_ds(:,2));
f33 = mldivide(ecog_3_train_R_ext.'*ecog_3_train_R_ext,ecog_3_train_R_ext.'*glove_3_train_ds(:,3));
f34 = mldivide(ecog_3_train_R_ext.'*ecog_3_train_R_ext,ecog_3_train_R_ext.'*glove_3_train_ds(:,4));
f35 = mldivide(ecog_3_train_R_ext.'*ecog_3_train_R_ext,ecog_3_train_R_ext.'*glove_3_train_ds(:,5));

%% Lasso regression for s1
[lasso_s11, inf11] = lasso(ecog_1_train_R_ext, glove_1_train_ds(:,1),'CV',10);
[lasso_s12, inf12] = lasso(ecog_1_train_R_ext, glove_1_train_ds(:,2),'CV',10);
[lasso_s13, inf13] = lasso(ecog_1_train_R_ext, glove_1_train_ds(:,3),'CV',10);
[lasso_s15, inf15] = lasso(ecog_1_train_R_ext, glove_1_train_ds(:,5),'CV',10);

%% Lasso regression for s2
[lasso_s21, inf21] = lasso(ecog_2_train_R_ext, glove_2_train_ds(:,1),'CV',10);
[lasso_s22, inf22] = lasso(ecog_2_train_R_ext, glove_2_train_ds(:,2),'CV',10);
[lasso_s23, inf23] = lasso(ecog_2_train_R_ext, glove_2_train_ds(:,3),'CV',10);
[lasso_s25, inf25] = lasso(ecog_2_train_R_ext, glove_2_train_ds(:,5),'CV',10);

%% Lasso regression for s3
[lasso_s31, inf31] = lasso(ecog_3_train_R_ext, glove_3_train_ds(:,1),'CV',10);
[lasso_s32, inf32] = lasso(ecog_3_train_R_ext, glove_3_train_ds(:,2),'CV',10);
[lasso_s33, inf33] = lasso(ecog_3_train_R_ext, glove_3_train_ds(:,3),'CV',10);
[lasso_s35, inf35] = lasso(ecog_3_train_R_ext, glove_3_train_ds(:,5),'CV',10);

%% Compute predictions from lasso model
p11 = ecog_1_test_R_ext*lasso_s11(:, inf11.Index1SE);
p12 = ecog_1_test_R_ext*lasso_s12(:, inf12.Index1SE);
p13 = ecog_1_test_R_ext*lasso_s13(:, inf13.Index1SE);
p15 = ecog_1_test_R_ext*lasso_s15(:, inf15.Index1SE);


p21 = ecog_2_test_R_ext*lasso_s21(:, inf21.Index1SE);
p22 = ecog_2_test_R_ext*lasso_s22(:, inf22.Index1SE);
p23 = ecog_2_test_R_ext*lasso_s23(:, inf23.Index1SE);
p25 = ecog_2_test_R_ext*lasso_s25(:, inf25.Index1SE);


p31 = ecog_3_test_R_ext*lasso_s31(:, inf31.Index1SE);
p32 = ecog_3_test_R_ext*lasso_s32(:, inf32.Index1SE);
p33 = ecog_3_test_R_ext*lasso_s33(:, inf33.Index1SE);
p35 = ecog_3_test_R_ext*lasso_s35(:, inf35.Index1SE);

%% Train a neural net
% net = fitnet(15);
% [net_s11,tr] = train(net,ecog_1_train_R_ext.',glove_1_train_ds(:,1).');
% nntraintool
% nntraintool('close')
% [net_s12,tr] = train(net,ecog_1_train_R_ext.',glove_1_train_ds(:,2).');
% nntraintool
% nntraintool('close')
% [net_s13,tr] = train(net,ecog_1_train_R_ext.',glove_1_train_ds(:,3).');
% nntraintool
% nntraintool('close')
% [net_s15,tr] = train(net,ecog_1_train_R_ext.',glove_1_train_ds(:,5).');
% nntraintool
% nntraintool('close')
% 
% [net_s21,tr] = train(net,ecog_2_train_R_ext.',glove_2_train_ds(:,1).');
% nntraintool
% nntraintool('close')
% [net_s22,tr] = train(net,ecog_2_train_R_ext.',glove_2_train_ds(:,2).');
% nntraintool
% nntraintool('close')
% [net_s23,tr] = train(net,ecog_2_train_R_ext.',glove_2_train_ds(:,3).');
% nntraintool
% nntraintool('close')
% [net_s25,tr] = train(net,ecog_2_train_R_ext.',glove_2_train_ds(:,5).');
% nntraintool
% nntraintool('close')
% 
% [net_s31,tr] = train(net,ecog_3_train_R_ext.',glove_3_train_ds(:,1).');
% nntraintool
% nntraintool('close')
% [net_s32,tr] = train(net,ecog_3_train_R_ext.',glove_3_train_ds(:,2).');
% nntraintool
% nntraintool('close')
% [net_s33,tr] = train(net,ecog_3_train_R_ext.',glove_3_train_ds(:,3).');
% nntraintool
% nntraintool('close')
% [net_s35,tr] = train(net,ecog_3_train_R_ext.',glove_3_train_ds(:,5).');
% nntraintool
% nntraintool('close')

%% Produce predictions for linear regression
% ecog_1_test_R_ext = [ecog_1_test_R;zeros(1,length(ecog_1_test_R(1,:)))];
% ecog_2_test_R_ext = [ecog_2_test_R;zeros(1,length(ecog_2_test_R(1,:)))];
% ecog_3_test_R_ext = [ecog_3_test_R;zeros(1,length(ecog_3_test_R(1,:)))];
% 
% p11 = ecog_1_test_R_ext*f11;
% p12 = ecog_1_test_R_ext*f12;
% p13 = ecog_1_test_R_ext*f13;
% p14 = ecog_1_test_R_ext*f14;
% p15 = ecog_1_test_R_ext*f15;
% 
% p21 = ecog_2_test_R_ext*f21;
% p22 = ecog_2_test_R_ext*f22;
% p23 = ecog_2_test_R_ext*f23;
% p24 = ecog_2_test_R_ext*f24;
% p25 = ecog_2_test_R_ext*f25;
%  
% p31 = ecog_3_test_R_ext*f31;
% p32 = ecog_3_test_R_ext*f32;
% p33 = ecog_3_test_R_ext*f33;
% p34 = ecog_3_test_R_ext*f34;
% p35 = ecog_3_test_R_ext*f35;
% %%
% p11 = net_s11(ecog_1_test_R_ext);
% p12 = net_s12(ecog_1_test_R_ext);
% p13 = net_s13(ecog_1_test_R_ext);
% p15 = net_s15(ecog_1_test_R_ext);
% 
% p21 = net_s21(ecog_2_test_R_ext);
% p22 = net_s22(ecog_2_test_R_ext);
% p23 = net_s23(ecog_2_test_R_ext);
% p25 = net_s25(ecog_2_test_R_ext);
% 
% p31 = net_s31(ecog_3_test_R_ext);
% p32 = net_s32(ecog_3_test_R_ext);
% p33 = net_s33(ecog_3_test_R_ext);
% p35 = net_s35(ecog_3_test_R_ext);



%% Interpolate predictions
x1 = linspace(1, length(p11),length(p11));
xq1 = linspace(1,length(p11),length(ecog_1_test(:,1)));

p11_full = interp1(x1,p11.',xq1);
p12_full = interp1(x1,p12.',xq1);
p13_full = interp1(x1,p13.',xq1);
p15_full = interp1(x1,p15.',xq1);

x2 = linspace(1,length(p21),length(p21));
xq2 = linspace(1,length(p21),length(ecog_2_test(:,1)));

p21_full = interp1(x2,p21.',xq2);
p22_full = interp1(x2,p22.',xq2);
p23_full = interp1(x2,p23.',xq2);
p25_full = interp1(x2,p25.',xq2);

x3 = linspace(1,length(p31),length(p31));
xq3 = linspace(1,length(p31),length(ecog_3_test(:,1)));

p31_full = interp1(x3,p31.',xq3);
p32_full = interp1(x3,p32.',xq3);
p33_full = interp1(x3,p33.',xq3);
p35_full = interp1(x3,p35.',xq3);
%% Produce dummy predictions for finger 4

p14_full = zeros(147500,1).';
p24_full = zeros(147500,1).';
p34_full = zeros(147500,1).';

%% Package predictions for submission
s1_preds = [p11_full.' p12_full.' p13_full.' p14_full.' p15_full.'];
s2_preds = [p21_full.' p22_full.' p23_full.' p24_full.' p25_full.'];
s3_preds = [p31_full.' p32_full.' p33_full.' p34_full.' p35_full.'];
predicted_dg = {s1_preds, s2_preds, s3_preds};