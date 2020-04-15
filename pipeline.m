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
%% Filter Function
% We used common average referencing (CAR) and a low pass filter after
% applying CAR with a cutoff at 60hz.
test_filtered = filter_data(ecog_1_train);
hold on;
plot(test_filtered(1:100,1));
plot(ecog_1_train(1:100,1));
legend("filtered", "regular");
hold off;
%% Get Features
% run getWindowedFeats_release function
test_feats = getWindowedFeats(ecog_1_train(1:1000,:), 1000, .1, .05);
% ecog_1_train_feats = getWindowedFeats(ecog_1_train, 1000, .1, .05);
% ecog_1_train_feats = getWindowedFeats(ecog_1_train, 1000, .1, .05);
% ecog_2_train_feats = getWindowedFeats(ecog_2_train, 1000, .1, .05);
% ecog_3_train_feats = getWindowedFeats(ecog_3_train, 1000, .1, .05);
% 
% ecog_1_test_feats = getWindowedFeats(ecog_1_test, 1000, .1, .05);
% ecog_2_test_feats = getWindowedFeats(ecog_2_test, 1000, .1, .05);
% ecog_3_test_feats = getWindowedFeats(ecog_3_test, 1000, .1, .05);
% The number of time bins/time windows is 3999 (the expression 
% to calculate this is in getWindowedFeats function and stored in the
% variable "num_windows"
%% Create R matrix

% In response to the question:
% The dimensions of the R_matrix would be (3999x(3*62*4)), assuming
% that the number of features used is 4.

ecog_1_train_R = create_R_matrix(ecog_1_train_feats, 3);
ecog_2_train_R = create_R_matrix(ecog_2_train_feats, 3);
ecog_3_train_R = create_R_matrix(ecog_3_train_feats, 3);
ecog_1_test_R = create_R_matrix(ecog_1_test_feats, 3);
ecog_2_test_R = create_R_matrix(ecog_2_test_feats, 3);
ecog_3_test_R = create_R_matrix(ecog_3_test_feats, 3);

test = create_R_matrix(testR_features, 3);
mean(mean(test))
%% Train classifiers (8 points)
% Classifier 1: Get angle predictions using optimal linear decoding. That is, 
% calculate the linear filter (i.e. the weights matrix) as defined by 
% Equation 1 for all 5 finger angles.

% downsample the gloves

chunk_sz = floor(length(glove_1_train(:,1))/length(ecog_1_train_R(:,1)));
num_chunks = length(glove_1_train(:,1))/chunk_sz;
glove_1_train_ds = zeros(num_chunks,5);
glove_2_train_ds = zeros(num_chunks,5);
glove_3_train_ds = zeros(num_chunks,5);

ecog_1_train_R_ext = [ecog_1_train_R;zeros(1,length(ecog_1_train_R(1,:)))];
ecog_2_train_R_ext = [ecog_2_train_R;zeros(1,length(ecog_2_train_R(1,:)))];
ecog_3_train_R_ext = [ecog_3_train_R;zeros(1,length(ecog_3_train_R(1,:)))];

%downsample training glove data
for i = 0:num_chunks-1
    chunk1 = glove_1_train(i*chunk_sz+1:i*chunk_sz+chunk_sz,:);
    chunk2 = glove_2_train(i*chunk_sz+1:i*chunk_sz+chunk_sz,:);
    chunk3 = glove_3_train(i*chunk_sz+1:i*chunk_sz+chunk_sz,:);
    for j = 1:length(glove_1_train(1,:))
        glove_1_train_ds(i+1,j) = mean(chunk1(:,j));
        glove_2_train_ds(i+1,j) = mean(chunk2(:,j));
        glove_3_train_ds(i+1,j) = mean(chunk3(:,j));
    end   
end

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

% downsample testing glove data
chunk_sz = 50; %(100000/2000)
num_chunks = length(glove_1_test(:,1))/chunk_sz;

glove_1_test_ds = zeros(num_chunks,5);
glove_2_test_ds = zeros(num_chunks,5);
glove_3_test_ds = zeros(num_chunks,5);

ecog_1_test_R_ext = [ecog_1_test_R;zeros(1,length(ecog_1_test_R(1,:)))];
ecog_2_test_R_ext = [ecog_2_test_R;zeros(1,length(ecog_2_test_R(1,:)))];
ecog_3_test_R_ext = [ecog_3_test_R;zeros(1,length(ecog_3_test_R(1,:)))];

for i = 0:num_chunks-1
    chunk1 = glove_1_test(i*chunk_sz+1:i*chunk_sz+chunk_sz,:);
    chunk2 = glove_2_test(i*chunk_sz+1:i*chunk_sz+chunk_sz,:);
    chunk3 = glove_3_test(i*chunk_sz+1:i*chunk_sz+chunk_sz,:);
    for j = 1:length(glove_1_test(1,:))
        glove_1_test_ds(i+1,j) = mean(chunk1(:,j));
        glove_2_test_ds(i+1,j) = mean(chunk2(:,j));
        glove_3_test_ds(i+1,j) = mean(chunk3(:,j));
    end   
end

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



%%
% Try at least 1 other type of machine learning algorithm, you may choose
% to loop through the fingers and train a separate classifier for angles 
% corresponding to each finger

% Since this is mainly ensuring that we are able to build the pipeline
% correctly, I am applying the below algorithm only to subject 1.
% Compute line length with over N = 20 preceeding time bins (inclusive)
%for a given time bin. This is the feature for that time bin that will
%be used for linear regression using matlab's fitlm function

ecog_1_train_ext = [ecog_1_train(1:19,:); ecog_1_train];
N = 20;
feature_mat = zeros(length(ecog_1_train(:,1)),length(ecog_1_train(1,:)));
base = 0;
for i = 1:length(ecog_1_train(:,1))
    chunk = ecog_1_train_ext(base+1:base+N - 1,:);
    feature_vec = zeros(1,length(feature_mat(1,:)));
    for j = 1:length(feature_mat(1,:))
        feature_vec(j) = sum(abs(diff(chunk(:,j))));
    end
    feature_mat(i,:) = feature_vec;
    base = base + 1;
end
%%
% linear regressions over line length feature mat for prediction
mdl1 = fitlm(feature_mat, glove_1_train(:,1));
mdl2 = fitlm(feature_mat, glove_1_train(:,2));
mdl3 = fitlm(feature_mat, glove_1_train(:,3));
mdl4 = fitlm(feature_mat, glove_1_train(:,4));
mdl5 = fitlm(feature_mat, glove_1_train(:,5));

p1 = mdl1.predict(ecog_1_test);
p2 = mdl2.predict(ecog_1_test);
p3 = mdl3.predict(ecog_1_test);
p4 = mdl4.predict(ecog_1_test);
p5 = mdl5.predict(ecog_1_test);

R1 = corrcoef(p1, glove_1_test(:,1));
R2 = corrcoef(p2, glove_1_test(:,2));
R3 = corrcoef(p3, glove_1_test(:,3));
R4 = corrcoef(p4, glove_1_test(:,4));
R5 = corrcoef(p5, glove_1_test(:,5));

scatter([1,2,3,4,5], [R1(1,2), R2(1,2), R3(1,2), R4(1,2), R5(1,2)])
title("R value for pred. & actual angles for subject 1 using line len")
xlabel("finger number")
ylabel("R value")
%%
% Try a form of either feature or prediction post-processing to try and
% improve underlying data or predictions.

% Trying to take the square of the line lengths in order to possibly accentuate
% differences to get more varied angle predictions

feature_mat_sq = feature_mat.^2;
mdl1 = fitlm(feature_mat_sq, glove_1_train(:,1));
mdl2 = fitlm(feature_mat_sq, glove_1_train(:,2));
mdl3 = fitlm(feature_mat_sq, glove_1_train(:,3));
mdl4 = fitlm(feature_mat_sq, glove_1_train(:,4));
mdl5 = fitlm(feature_mat_sq, glove_1_train(:,5));

p1 = mdl1.predict(ecog_1_test);
p2 = mdl2.predict(ecog_1_test);
p3 = mdl3.predict(ecog_1_test);
p4 = mdl4.predict(ecog_1_test);
p5 = mdl5.predict(ecog_1_test);

R1 = corrcoef(p1, glove_1_test(:,1));
R2 = corrcoef(p2, glove_1_test(:,2));
R3 = corrcoef(p3, glove_1_test(:,3));
R4 = corrcoef(p4, glove_1_test(:,4));
R5 = corrcoef(p5, glove_1_test(:,5));

scatter([1,2,3,4,5], [R1(1,2), R2(1,2), R3(1,2), R4(1,2), R5(1,2)])
title("R value for pred. & actual angles for subject 1 using squared line len")
xlabel("finger number")
ylabel("R value")


%% Correlate data to get test accuracy and make figures (2 point)

% Calculate accuracy by correlating predicted and actual angles for each
% finger separately. Hint: You will want to use zohinterp to ensure both 
% vectors are the same length.

% Doing this only for the optimal linear decoder since the processs
% is the very similar for the alternative classifier

R11 = corrcoef(p11, glove_1_test_ds(:,1));
R12 = corrcoef(p12, glove_1_test_ds(:,2));
R13 = corrcoef(p13, glove_1_test_ds(:,3));
R14 = corrcoef(p14, glove_1_test_ds(:,4));
R15 = corrcoef(p15, glove_1_test_ds(:,5));

R21 = corrcoef(p21, glove_2_test_ds(:,1));
R22 = corrcoef(p22, glove_2_test_ds(:,2));
R23 = corrcoef(p23, glove_2_test_ds(:,3));
R24 = corrcoef(p24, glove_2_test_ds(:,4));
R25 = corrcoef(p25, glove_2_test_ds(:,5));

R31 = corrcoef(p31, glove_3_test_ds(:,1));
R32 = corrcoef(p32, glove_3_test_ds(:,2));
R33 = corrcoef(p33, glove_3_test_ds(:,3));
R34 = corrcoef(p34, glove_3_test_ds(:,4));
R35 = corrcoef(p35, glove_3_test_ds(:,5));

subplot(2,2,1)
scatter([1,2,3,4,5], [R11(1,2), R12(1,2), R13(1,2), R14(1,2), R15(1,2)])
title("R value for pred. & actual angles for subject 1")
xlabel("finger number")
ylabel("R value")
subplot(2,2,2)
scatter([1,2,3,4,5], [R21(1,2), R22(1,2), R23(1,2), R24(1,2), R25(1,2)])
title("R value for pred. & actual angles for subject 2")
xlabel("finger number")
ylabel("R value")
subplot(2,2,3)
scatter([1,2,3,4,5], [R31(1,2), R32(1,2), R33(1,2), R34(1,2), R35(1,2)])
title("R value for pred. & actual angles for subject 3")
xlabel("finger number")
ylabel("R value")


