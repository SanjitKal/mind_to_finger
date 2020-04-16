% Pipeline.m - runs the predicion pipeline:
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

num_ch_1 = 61;
num_ch_2 = 46;
num_ch_3 = 64;

% num_ch_1 = 4; %Only use channels 18-22
% num_ch_2 = 4; %Only use channels 18-22
% num_ch_3 = 4; %Only use channels 18-22

% Split data into a train and test set (use at least 50% for training)
num_train = 200000;
num_test = 100000;

glove_1_train = glove_1{1}(1:200000,1:5);
glove_1_test = glove_1{1}(200001:300000,1:5);
ecog_1_train = ecog_1{1}(1:200000, 1:num_ch_1);
ecog_1_test = ecog_1{1}(200001:300000, 1:num_ch_1);
  
glove_2_train = glove_2{1}(1:200000,1:5);
glove_2_test = glove_2{1}(200001:300000,1:5);
ecog_2_train = ecog_2{1}(1:200000, 1:num_ch_2);
ecog_2_test = ecog_2{1}(200001:300000, 1:num_ch_2);
  
glove_3_train = glove_3{1}(1:200000,1:5);
glove_3_test = glove_3{1}(200001:300000,1:5);
ecog_3_train = ecog_3{1}(1:200000, 1:num_ch_3);
ecog_3_test = ecog_3{1}(200001:300000, 1:num_ch_3);

% glove_1_train = glove_1{1}(1:200000,1:5);
% glove_1_test = glove_1{1}(200001:300000,1:5);
% ecog_1_train = ecog_1{1}(1:200000, 18:22);
% ecog_1_test = ecog_1{1}(200001:300000, 18:22);
%  
% glove_2_train = glove_2{1}(1:200000,1:5);
% glove_2_test = glove_2{1}(200001:300000,1:5);
% ecog_2_train = ecog_2{1}(1:200000, 18:22);
% ecog_2_test = ecog_2{1}(200001:300000, 18:22);
%  
% glove_3_train = glove_3{1}(1:200000,1:5);
% glove_3_test = glove_3{1}(200001:300000,1:5);
% ecog_3_train = ecog_3{1}(1:200000, 18:22);
% ecog_3_test = ecog_3{1}(200001:300000, 18:22);
%% Filter Function
% We apply CAR (common average referencing) to smooth the data
test_filtered = filter_data(ecog_1_train);
%%
% plotting to make sure filter produces sensible results.
hold on;
plot(test_filtered(1:100,1));
plot(ecog_1_train(1:100,1));
legend("filtered ch 20", "regular ch 20");
hold off;
%%
% plotting to make sure filter produces sensible results.
hold on;
plot(test_filtered(1:100,2));
plot(ecog_1_train(1:100,2));
legend("filtered ch 22", "regular ch 22");
hold off;
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
    glove_3_train_ds(:,i) = decimate(glove_2_train(:,i), chunk_sz);
end
%%

ecog_1_train_R_ext = [ecog_1_train_R; ecog_1_train_R(length(ecog_1_train_R(:,1)),:)];
ecog_2_train_R_ext = [ecog_2_train_R; ecog_2_train_R(length(ecog_2_train_R(:,1)),:)];
ecog_3_train_R_ext = [ecog_3_train_R; ecog_3_train_R(length(ecog_3_train_R(:,1)),:)];

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

%%
ecog_1_test_R_ext = [ecog_1_test_R;zeros(1,length(ecog_1_test_R(1,:)))];
ecog_2_test_R_ext = [ecog_2_test_R;zeros(1,length(ecog_2_test_R(1,:)))];
ecog_3_test_R_ext = [ecog_3_test_R;zeros(1,length(ecog_3_test_R(1,:)))];

p11 = ecog_1_test_R_ext*f11;
p12 = ecog_1_test_R_ext*f12;
p13 = ecog_1_test_R_ext*f13;
p14 = ecog_1_test_R_ext*f14;
p15 = ecog_1_test_R_ext*f15;

p21 = ecog_2_test_R_ext*f21;
p22 = ecog_2_test_R_ext*f22;
p23 = ecog_2_test_R_ext*f23;
p24 = ecog_2_test_R_ext*f24;
p25 = ecog_2_test_R_ext*f25;
 
p31 = ecog_3_test_R_ext*f31;
p32 = ecog_3_test_R_ext*f32;
p33 = ecog_3_test_R_ext*f33;
p34 = ecog_3_test_R_ext*f34;
p35 = ecog_3_test_R_ext*f35;
%% Correlate data to get test accuracy and make figures (2 point)

% Calculate accuracy by correlating predicted and actual angles for each
% finger separately. Hint: You will want to use zohinterp to ensure both 
% vectors are the same length.
x1 = linspace(1, length(p11),length(p11));
xq1 = linspace(1,length(p11),length(glove_1_test(:,1)));

p11_full = spline(x1,[0 p11.' 0],xq1);
p12_full = spline(x1,[0 p12.' 0],xq1);
p13_full = spline(x1,[0 p13.' 0],xq1);
p15_full = spline(x1,[0 p15.' 0],xq1);

x2 = linspace(1,length(p21),length(p21));
xq2 = linspace(1,length(p21),length(glove_2_test(:,1)));

p21_full = spline(x2,[0 p21.' 0],xq2);
p22_full = spline(x2,[0 p22.' 0],xq2);
p23_full = spline(x2,[0 p23.' 0],xq2);
p25_full = spline(x2,[0 p25.' 0],xq2);

x3 = linspace(1,length(p31),length(p31));
xq3 = linspace(1,length(p31),length(glove_3_test(:,1)));

p31_full = spline(x3,[0 p31.' 0],xq3);
p32_full = spline(x3,[0 p32.' 0],xq3);
p33_full = spline(x3,[0 p33.' 0],xq3);
p35_full = spline(x3,[0 p35.' 0],xq3);
%%
R11 = corr(p11_full.', glove_1_test(:,1));
R12 = corr(p12_full.', glove_1_test(:,2));
R13 = corr(p13_full.', glove_1_test(:,3));
R15 = corr(p15_full.', glove_1_test(:,5));

R21 = corr(p21_full.', glove_2_test(:,1));
R22 = corr(p22_full.', glove_2_test(:,2));
R23 = corr(p23_full.', glove_2_test(:,3));
R25 = corr(p25_full.', glove_2_test(:,5));

R31 = corr(p31_full.', glove_3_test(:,1));
R32 = corr(p32_full.', glove_3_test(:,2));
R33 = corr(p33_full.', glove_3_test(:,3));
R35 = corr(p35_full.', glove_3_test(:,5));
%%
scatter([1,2,3,5], [R11, R12, R13, R15])
title("R value for pred. & actual angles for subject 1")
xlabel("finger number")
ylabel("R value")
%%
scatter([1,2,3,5], [R21, R22, R23, R25])
title("R value for pred. & actual angles for subject 2")
xlabel("finger number")
ylabel("R value")
%%
scatter([1,2,3,5], [R31, R32, R33, R35])
title("R value for pred. & actual angles for subject 3")
xlabel("finger number")
ylabel("R value")
%%
average_R1 = mean([R11, R12, R13, R15])
average_R2 = mean([R21, R22, R23, R25])
average_R3 = mean([R31, R32, R33, R35])
average_R = mean([average_R1, average_R2, average_R3])

